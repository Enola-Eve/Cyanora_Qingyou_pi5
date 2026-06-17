
# coding=utf-8
import os, cv2, time, json, socket, numpy as np, asyncio, threading, datetime, sys

# ================= 【树莓派适配】使用 tflite-runtime 替代 tensorflow =================
# 注意：你需要先把模型转换为 .tflite 格式
from tflite_runtime.interpreter import Interpreter

import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt, find_peaks
from bleak import BleakScanner, BleakClient
import customtkinter as ctk

# ================= 全局事件循环 (适配树莓派) =================
ble_event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(ble_event_loop)

# ================= 全局变量与初始化 =================
hr_lock = threading.Lock()
BROADCAST_PORT = 5000
broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

last_emotion = "未检测到"
last_confidence = 0.0
current_hr = 0.0
ble_heart_rate = 0.0
ble_client_global = None
last_heart_rate_time = 0

HEART_RATE_MEASUREMENT_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

print("正在加载模型...")
# --- 模型加载方式已修改 ---
try:
    # 1. 加载 TFLite 模型
    interpreter = Interpreter(model_path='emotion_model.tflite')
    interpreter.allocate_tensors()

    # 2. 获取输入输出张量索引
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Calm'}
    print(" 模型加载成功 ")
except Exception as e:
    print(f" 模型加载失败: {e}")
    sys.exit()

try:
    font = ImageFont.truetype("simhei.ttf", 30)
except IOError:
    font = ImageFont.load_default()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

BUFFER_SIZE = 300
rgb_buffer = []
timestamps = []


# ================= 蓝牙核心逻辑 =================
def handle_hr_data(sender, data):
    global ble_heart_rate, last_heart_rate_time
    try:
        flags = data[0]
        if flags & 0x01:
            hr = int.from_bytes(data[1:3], byteorder='little')
        else:
            hr = data[1]
        with hr_lock:
            ble_heart_rate = float(hr)
            last_heart_rate_time = time.time()
        print(f"✅ 解析后心率: {hr} BPM")
    except Exception as e:
        print(f"⚠️ 心率解析异常: {e}")


async def scan_ble_devices():
    print("⏳ 正在扫描周围所有蓝牙设备（耗时约5秒）...")
    devices = await BleakScanner.discover(timeout=5.0)
    print(f"📡 共发现 {len(devices)} 个蓝牙设备：")
    hr_devices = []
    target_uuid = "0000180d-0000-1000-8000-00805f9b34fb"
    for d in devices:
        try:
            print(f" 👉 名称: {d.name} | MAC: {d.address}")
            name = (d.name or "").lower()
            if any(keyword in name for keyword in ["heart", "hr", "band", "watch", "mi band", "huawei"]):
                hr_devices.append((d.address, d.name or "Unknown Device"))
                continue
            adv_data = getattr(d.details, 'advertisement_data', None)
            if adv_data and hasattr(adv_data, 'service_uuids'):
                if target_uuid in adv_data.service_uuids:
                    hr_devices.append((d.address, d.name or "Unknown Device"))
                    continue
        except Exception as e:
            continue
    return hr_devices


async def connect_ble_device(address):
    global ble_client_global, ble_heart_rate
    if ble_client_global and ble_client_global.is_connected:
        await ble_client_global.disconnect()
    try:
        client = BleakClient(address)
        await client.connect()
        print(f"🔗 成功连接到: {address}")
        print("📜 正在读取设备所有服务...")
        target_char_uuid = None
        for service in client.services:
            print(f" 📦 服务: {service.description} ({service.uuid})")
            for char in service.characteristics:
                props = ", ".join(char.properties)
                print(f" 🔹 特征: {char.description} ({char.uuid}) | 属性: [{props}]")
                if ("notify" in props or "indicate" in props) and "heart" in char.description.lower():
                    target_char_uuid = char.uuid
        subscribe_uuid = target_char_uuid or HEART_RATE_MEASUREMENT_CHAR_UUID
        try:
            await client.start_notify(subscribe_uuid, handle_hr_data)
            ble_client_global = client
            print(f"✅ 订阅成功！UUID: {subscribe_uuid}")
            vendor_char_uuid = "00004a02-0000-1000-8000-00805f9b34fb"
            activate_commands = [bytearray([0x01]), bytearray([0x13, 0x01])]
            for cmd in activate_commands:
                try:
                    await client.write_gatt_char(vendor_char_uuid, cmd, response=False)
                    print(f"📤 激活指令发送: {cmd.hex()}")
                    await asyncio.sleep(0.5)
                except:
                    pass
            print("🚀 请在手表上开始测心率，等待数据...")
            return True
        except Exception as e:
            print(f"❌ 订阅失败: {e}")
            return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


async def safe_disconnect_ble():
    global ble_client_global
    if ble_client_global:
        try:
            if ble_client_global.is_connected:
                await ble_client_global.stop_notify(HEART_RATE_MEASUREMENT_CHAR_UUID)
                await ble_client_global.disconnect()
        except Exception as e:
            print(f"断开蓝牙时出现警告(可忽略): {e}")
        finally:
            ble_client_global = None


# ================= 摄像头测心率备用逻辑 =================
def get_camera_heart_rate():
    if len(rgb_buffer) < BUFFER_SIZE:
        return 0.0
    signal = np.array(rgb_buffer)[:, 1]
    b, a = butter(4, [0.7, 4.0], fs=30.0, btype='bandpass')
    filtered_signal = filtfilt(b, a, signal)
    peaks, _ = find_peaks(filtered_signal, distance=10)
    if len(peaks) < 2:
        return 0.0
    rr_intervals = np.diff(np.array(timestamps)[peaks])
    avg_rr = np.mean(rr_intervals)
    return 60.0 / avg_rr if avg_rr > 0 else 0.0


# ================= 现代化图形界面 (GUI) =================
class BluetoothManagerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("智能表情与心率广播系统")
        self.geometry("400x500")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        ctk.CTkLabel(self, text="BLE 心率广播管理器", font=("Segoe UI", 20, "bold")).pack(pady=20)
        self.scan_btn = ctk.CTkButton(self, text="🔍 扫描附近手表", command=self.start_scan_thread, height=40)
        self.scan_btn.pack(pady=10, padx=20, fill="x")
        self.device_listbox = ctk.CTkTextbox(self, height=200, state="disabled")
        self.device_listbox.pack(pady=10, padx=20, fill="both", expand=True)
        self.connect_btn = ctk.CTkButton(self, text="🔗 连接选中设备", command=self.connect_selected, height=40)
        self.connect_btn.pack(pady=10, padx=20, fill="x")
        self.status_label = ctk.CTkLabel(self, text="状态: 就绪", text_color="grey")
        self.status_label.pack(pady=10)
        self.scanned_devices = []
        self.cap = None
        self.video_running = False

    def start_scan_thread(self):
        threading.Thread(target=self.perform_scan, daemon=True).start()

    def perform_scan(self):
        self.scan_btn.configure(state="disabled", text="扫描中...")
        self.status_label.configure(text="正在搜索心率广播设备...", text_color="orange")
        devices = ble_event_loop.run_until_complete(scan_ble_devices())
        self.scanned_devices = devices
        self.device_listbox.configure(state="normal")
        self.device_listbox.delete("1.0", "end")
        if devices:
            for i, (addr, name) in enumerate(devices):
                self.device_listbox.insert("end", f"{i + 1}. {name}\n MAC: {addr}\n\n")
            self.status_label.configure(text=f"发现 {len(devices)} 个设备", text_color="green")
        else:
            self.device_listbox.insert("end", "未发现支持心率广播的设备。\n请确保手表已开启该功能！")
            self.status_label.configure(text="未找到设备", text_color="red")
        self.device_listbox.configure(state="disabled")
        self.scan_btn.configure(state="normal", text="🔍 重新扫描")

    def connect_selected(self):
        try:
            selected = self.device_listbox.get("sel.first", "sel.last").strip()
        except:
            self.status_label.configure(text="请先在列表中点击选中一个设备！", text_color="red")
            return
        if not selected:
            self.status_label.configure(text="请先选中一个设备！", text_color="red")
            return
        mac_address = selected.split("MAC: ")[-1].strip()
        threading.Thread(target=self.perform_connect, args=(mac_address,), daemon=True).start()

    def perform_connect(self, mac):
        self.connect_btn.configure(state="disabled")
        self.status_label.configure(text="正在连接...", text_color="orange")
        success = ble_event_loop.run_until_complete(connect_ble_device(mac))
        if success:
            self.status_label.configure(text="已连接并接收心率数据！", text_color="green")
            self.start_camera()
        else:
            self.status_label.configure(text="连接失败，请重试", text_color="red")
            self.connect_btn.configure(state="normal")

    def start_camera(self):
        if not self.video_running:
            self.video_running = True
            threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        self.cap = cv2.VideoCapture(0)
        last_analysis_time = time.time()
        try:
            pil_font = ImageFont.truetype("simhei.ttf", 30)
        except IOError:
            pil_font = ImageFont.load_default()

        while self.video_running:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            current_time = time.time()
            rgb_data = None
            face_box = None

            # --- 1. 视频帧预处理与人脸检测 ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                face_box = (x_min, y_min, x_max - x_min, y_max - y_min)
                try:
                    x1 = int(landmarks[109].x * w);
                    y1 = int(landmarks[10].y * h)
                    x2 = int(landmarks[338].x * w);
                    y2 = int(landmarks[152].y * h)
                    roi = rgb_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if roi.size > 0:
                        rgb_data = np.mean(np.mean(roi, axis=0), axis=0)
                except:
                    pass
                if face_box:
                    x, y, w, h = face_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # --- 2. 情绪识别 (每0.3秒一次) ---
            if current_time - last_analysis_time > 0.3:
                if face_box:
                    try:
                        x, y, w, h = face_box
                        face_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
                        if face_img.size > 0:
                            gray_face = cv2.resize(face_img, (48, 48)) / 255.0

                            # --- TFLite 推理方式已修改 ---
                            input_tensor = np.expand_dims(np.expand_dims(gray_face, 0), -1).astype(np.float32)
                            interpreter.set_tensor(input_details[0]['index'], input_tensor)
                            interpreter.invoke()
                            preds = interpreter.get_tensor(output_details[0]['index'])[0]

                            global last_emotion, last_confidence
                            last_emotion = emotion_labels[np.argmax(preds)]
                            last_confidence = np.max(preds)
                    except Exception as e:
                        print(f"情绪识别错误: {e}")
                last_analysis_time = current_time

            # --- 3. 心率数据融合 (带超时检测) ---
            global current_hr, ble_heart_rate, last_heart_rate_time
            hr_source = "Camera"
            with hr_lock:
                safe_ble_hr = ble_heart_rate
                # 检查：如果当前时间 - 最后收到时间 > 3秒，视为断开
                if time.time() - last_heart_rate_time > 3.0:
                    safe_ble_hr = 0.0

            if safe_ble_hr > 0:
                current_hr = safe_ble_hr
                hr_source = "BLE Watch"
            else:
                if rgb_data is not None:
                    rgb_buffer.append(rgb_data)
                    timestamps.append(current_time)
                    if len(rgb_buffer) > BUFFER_SIZE:
                        rgb_buffer.pop(0)
                        timestamps.pop(0)
                    current_hr = get_camera_heart_rate()

            # --- 4. 画面绘制 (使用 PIL) ---
            try:
                if frame is None or frame.size == 0:
                    continue
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                # 【修改】固定显示指定的时间和地点
                # now = datetime.datetime.now()
                # time_str = now.strftime("%Y-%m-%d") + " 星期" + "一二三四五六日"[now.weekday()]
                time_str = "2026-06-16 星期二"
                location_str = "天津市 天津市"

                draw.text((10, 10), f"Emotion: {last_emotion} ({last_confidence:.2f})", font=pil_font,
                          fill=(255, 255, 0))
                draw.text((10, 50), f"HR: {current_hr:.1f} BPM [{hr_source}]", font=pil_font, fill=(0, 255, 0))
                draw.text((10, 90), f"{time_str} | {location_str}", font=pil_font, fill=(255, 255, 255))
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"画面绘制出错: {e}")
                continue

            # --- 5. UDP 广播与窗口显示 ---
            try:
                broadcast_data = json.dumps({
                    "emotion": last_emotion,
                    "heart_rate": round(current_hr, 1),
                    "time": time_str,
                    "location": location_str
                })
                broadcast_socket.sendto(broadcast_data.encode('utf-8'), ('<broadcast>', BROADCAST_PORT))
            except:
                pass

            cv2.imshow("Emotion & Heart Rate Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def destroy(self):
        self.video_running = False
        time.sleep(0.2)
        global ble_client_global, ble_event_loop
        if ble_client_global and not ble_event_loop.is_closed():
            try:
                ble_event_loop.run_until_complete(safe_disconnect_ble())
            except Exception as e:
                print(f"断开蓝牙时出现警告(可忽略): {e}")
        if not ble_event_loop.is_closed():
            ble_event_loop.close()
        super().destroy()


if __name__ == "__main__":
    app = BluetoothManagerApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("\n⚠️ 检测到用户中断，正在安全退出...")
        app.destroy()
    except Exception as e:
        print(f"\n❌ 程序发生未知错误: {e}")
    finally:
        print(" 程序已安全退出。")