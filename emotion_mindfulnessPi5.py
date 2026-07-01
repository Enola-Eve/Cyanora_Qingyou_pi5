# coding=utf-8
import os, cv2, time, json, socket, numpy as np, asyncio, threading, datetime, sys, math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
from scipy.signal import butter, filtfilt, find_peaks
from bleak import BleakScanner, BleakClient

# ================= 1. 全局变量与初始化 =================
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
ble_connected = False
HEART_RATE_MEASUREMENT_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# ⚠️ 注意：请修改为你的模型在树莓派上的实际路径
model_path = '/home/pi/simple_CNN.530-0.65.hdf5'

print(f"正在加载模型: {model_path}")
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
try:
    emotion_classifier = tf.keras.models.load_model(model_path)
    emotion_labels = {0: '生气', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '难过', 5: '惊喜', 6: '平静'}
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 情绪模型加载失败: {e}")
    sys.exit()

try:
    font = ImageFont.truetype("Arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

mp_face_mesh = tf.keras.utils.get_file('face_mesh.task',
                                       'https://storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker.task')  # Placeholder, using old mediapipe for now
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

BUFFER_SIZE = 300
rgb_buffer = []
timestamps = []


# ================= 2. 蓝牙核心逻辑 =================
def handle_hr_data(sender, data):
    global ble_heart_rate, last_heart_rate_time
    try:
        flags = data[0]
        hr = int.from_bytes(data[1:3], byteorder='little') if flags & 0x01 else data[1]
        with hr_lock:
            ble_heart_rate = float(hr)
            last_heart_rate_time = time.time()
    except Exception as e:
        print(f"⚠️ 心率解析异常: {e}")


async def scan_ble_devices():
    devices = await BleakScanner.discover(timeout=5.0)
    hr_devices = []
    target_uuid = "0000180d-0000-1000-8000-00805f9b34fb"
    for d in devices:
        try:
            name = (d.name or "").lower()
            if any(keyword in name for keyword in ["heart", "hr", "band", "watch", "mi band", "huawei"]):
                hr_devices.append((d.address, d.name or "Unknown Device"))
                continue
            adv_data = getattr(d.details, 'advertisement_data', None)
            if adv_data and hasattr(adv_data, 'service_uuids'):
                if target_uuid in adv_data.service_uuids:
                    hr_devices.append((d.address, d.name or "Unknown Device"))
        except:
            continue
    return hr_devices


async def connect_ble_device(address):
    global ble_client_global, ble_connected
    if ble_client_global and ble_client_global.is_connected:
        await ble_client_global.disconnect()
    try:
        client = BleakClient(address)
        await client.connect()
        target_char_uuid = None
        for service in client.services:
            for char in service.characteristics:
                props = ", ".join(char.properties)
                if ("notify" in props or "indicate" in props) and "heart" in char.description.lower():
                    target_char_uuid = char.uuid

        subscribe_uuid = target_char_uuid or HEART_RATE_MEASUREMENT_CHAR_UUID
        try:
            await client.start_notify(subscribe_uuid, handle_hr_data)
            ble_client_global = client
            with hr_lock:
                ble_connected = True
            return True
        except Exception as e:
            print(f"❌ 订阅失败: {e}")
            return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


# ================= 3. 摄像头测心率逻辑 =================
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


# ================= 4. 正念呼吸引导窗口 (Tkinter版) =================
class MindfulnessWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("🌊 正念呼吸引导")
        self.geometry("500x550")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=500, height=450, bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.breath_label = tk.Label(self, text="准备开始...", font=("Arial", 24, "bold"), fg="white", bg="#1a1a2e")
        self.breath_label.pack(pady=10)

        self.cx, self.cy = 250, 220
        self.start_time = time.time()
        self.animate()

    def draw_flower(self, emotion, scale, wave_progress):
        self.canvas.delete("all")

        # 1. 绘制海浪波纹
        max_wave_radius = 200
        current_wave_radius = max_wave_radius * wave_progress
        if current_wave_radius > 10:
            wave_color = "#87CEFA"
            line_width = max(0, 4 * (1 - wave_progress))
            self.canvas.create_oval(
                self.cx - current_wave_radius, self.cy - current_wave_radius,
                self.cx + current_wave_radius, self.cy + current_wave_radius,
                outline=wave_color, width=line_width
            )

        # 2. 创建Pillow图像绘制花朵
        flower_size = 400
        flower_img = Image.new('RGBA', (flower_size, flower_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(flower_img)
        center = flower_size // 2

        # 3. 根据情绪决定花瓣数量
        if emotion == "开心":
            petal_count = 6
        elif emotion in ["难过", "生气"]:
            petal_count = 8
        else:
            petal_count = 5

        # 4. 绘制多层半透明花瓣
        base_radius = 140 * scale
        petal_width = 40 * scale
        for i in range(petal_count):
            angle = 2 * math.pi * i / petal_count
            px = center + base_radius * math.cos(angle)
            py = center + base_radius * math.sin(angle)
            draw.ellipse([px - petal_width, py - petal_width, px + petal_width, py + petal_width],
                         fill=(255, 191, 0, 120), outline=(255, 215, 0, 150))

        inner_radius = 80 * scale
        inner_width = 25 * scale
        for i in range(petal_count):
            angle = 2 * math.pi * (i + 0.5) / petal_count
            px = center + inner_radius * math.cos(angle)
            py = center + inner_radius * math.sin(angle)
            draw.ellipse([px - inner_width, py - inner_width, px + inner_width, py + inner_width],
                         fill=(255, 165, 0, 160), outline=(255, 200, 0, 180))

        # 5. 绘制花蕊和五角星
        heart_radius = 35 * scale
        draw.ellipse([center - heart_radius * 1.5, center - heart_radius * 1.5,
                      center + heart_radius * 1.5, center + heart_radius * 1.5],
                     fill=(255, 140, 0, 80))
        draw.ellipse([center - heart_radius, center - heart_radius,
                      center + heart_radius, center + heart_radius],
                     fill=(255, 100, 0, 240), outline=(255, 220, 100, 255))

        star_radius = 12 * scale
        star_points = []
        for i in range(5):
            outer_angle = math.pi / 2 + 2 * math.pi * i / 5
            star_points.append((center + star_radius * math.cos(outer_angle),
                                center - star_radius * math.sin(outer_angle)))
            inner_angle = outer_angle + math.pi / 5
            star_points.append((center + star_radius * 0.4 * math.cos(inner_angle),
                                center - star_radius * 0.4 * math.sin(inner_angle)))
        draw.polygon(star_points, fill=(255, 255, 200, 255))

        # 6. 显示图像
        self.imgtk = ImageTk.PhotoImage(flower_img)
        self.canvas.create_image(self.cx, self.cy, image=self.imgtk, anchor="center")

    def animate(self):
        global last_emotion
        elapsed = time.time() - self.start_time
        breath_val = math.sin(elapsed * math.pi / 4)
        scale = 1.0 + 0.5 * breath_val
        wave_progress = (breath_val + 1) / 2.0

        if breath_val > 0.1:
            self.breath_label.configure(text="吸气...")
        elif breath_val < -0.1:
            self.breath_label.configure(text="呼气...")
        else:
            self.breath_label.configure(text="保持...")

        self.draw_flower(last_emotion, scale, wave_progress)
        self.after(40, self.animate)


# ================= 5. 主程序界面 (Tkinter版) =================
class BluetoothManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BLE 心率广播管理器")
        self.geometry("400x650")

        tk.Label(self, text="BLE 心率广播管理器", font=("Arial", 20, "bold")).pack(pady=10)

        self.video_label = tk.Label(self)
        self.video_label.pack(pady=10, padx=20)

        self.scan_btn = tk.Button(self, text="🔍 扫描附近手表", command=self.start_scan_thread, height=2)
        self.scan_btn.pack(pady=5, padx=20, fill="x")

        self.device_listbox = tk.Text(self, height=6, state="disabled")
        self.device_listbox.pack(pady=5, padx=20, fill="both", expand=True)

        self.connect_btn = tk.Button(self, text="🔗 连接选中设备", command=self.connect_selected, height=2)
        self.connect_btn.pack(pady=5, padx=20, fill="x")

        self.use_camera_hr = tk.BooleanVar(value=True)
        self.hr_source_switch = tk.Checkbutton(self, text="使用摄像头测心率", variable=self.use_camera_hr,
                                               command=self.toggle_hr_source)
        self.hr_source_switch.pack(pady=10, padx=20, fill="x")

        self.status_label = tk.Label(self, text="状态: 就绪", fg="grey")
        self.status_label.pack(pady=5)

        self.scanned_devices = []
        self.cap = None
        self.video_running = False

    def toggle_hr_source(self):
        source = "摄像头" if self.use_camera_hr.get() else "蓝牙手表"
        self.status_label.configure(text=f"心率源已切换至: {source}", fg="orange")

    def start_scan_thread(self):
        threading.Thread(target=self.perform_scan, daemon=True).start()

    def perform_scan(self):
        self.scan_btn.configure(state="disabled", text="扫描中...")
        self.status_label.configure(text="正在搜索心率广播设备...", fg="orange")
        devices = asyncio.run(scan_ble_devices())
        self.scanned_devices = devices
        self.device_listbox.configure(state="normal")
        self.device_listbox.delete("1.0", "end")
        if devices:
            for i, (addr, name) in enumerate(devices):
                self.device_listbox.insert("end", f"{i + 1}. {name}\n MAC: {addr}\n\n")
            self.status_label.configure(text=f"发现 {len(devices)} 个设备", fg="green")
        else:
            self.device_listbox.insert("end", "未发现支持心率广播的设备。")
            self.status_label.configure(text="未找到设备", fg="red")
        self.device_listbox.configure(state="disabled")
        self.scan_btn.configure(state="normal", text="🔍 重新扫描")

    def connect_selected(self):
        try:
            selected = self.device_listbox.get("sel.first", "sel.last").strip()
        except:
            self.status_label.configure(text="请先在列表中点击选中一个设备！", fg="red")
            return
        if not selected:
            self.status_label.configure(text="请先选中一个设备！", fg="red")
            return
        mac_address = selected.split("MAC: ")[-1].strip()
        threading.Thread(target=self.perform_connect, args=(mac_address,), daemon=True).start()

    def perform_connect(self, mac):
        self.connect_btn.configure(state="disabled")
        self.status_label.configure(text="正在连接...", fg="orange")
        success = asyncio.run(connect_ble_device(mac))
        if success:
            self.status_label.configure(text="已连接蓝牙设备，等待数据...", fg="green")
            self.start_camera()
        else:
            self.status_label.configure(text="连接失败，请重试", fg="red")
        self.connect_btn.configure(state="normal")

    def start_camera(self):
        if not self.video_running:
            self.video_running = True
            threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        global last_emotion, last_confidence, current_hr, ble_heart_rate
        self.cap = cv2.VideoCapture(0)
        last_analysis_time = time.time()
        while self.video_running:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            current_time = time.time()
            rgb_data = None
            face_box = None
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

            if current_time - last_analysis_time > 0.3:
                if face_box:
                    try:
                        x, y, w, h = face_box
                        face_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
                        if face_img.size > 0:
                            gray_face = cv2.resize(face_img, (48, 48)) / 255.0
                            preds = emotion_classifier.predict(np.expand_dims(np.expand_dims(gray_face, 0), -1),
                                                               verbose=0)
                            last_emotion = emotion_labels[np.argmax(preds[0])]
                            last_confidence = np.max(preds[0])
                    except Exception as e:
                        print(f"情绪识别错误: {e}")
                last_analysis_time = current_time

            hr_source = "Camera"
            if self.use_camera_hr.get():
                if rgb_data is not None:
                    rgb_buffer.append(rgb_data)
                    timestamps.append(current_time)
                    if len(rgb_buffer) > BUFFER_SIZE:
                        rgb_buffer.pop(0)
                        timestamps.pop(0)
                    if len(rgb_buffer) >= BUFFER_SIZE:
                        current_hr = get_camera_heart_rate()
            else:
                with hr_lock:
                    if ble_connected and (current_time - last_heart_rate_time <= 3.0):
                        current_hr = ble_heart_rate
                        hr_source = "BLE Watch"
                    else:
                        hr_source = "BLE (No Data)"
                if rgb_data is not None:
                    rgb_buffer.append(rgb_data)
                    timestamps.append(current_time)
                    if len(rgb_buffer) > BUFFER_SIZE:
                        rgb_buffer.pop(0)
                        timestamps.pop(0)

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb_frame)
                draw = ImageDraw.Draw(img_pil)
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d") + " 星期" + "一二三四五六日"[now.weekday()]
                draw.text((10, 10), f"Emotion: {last_emotion} ({last_confidence:.2f})", font=font, fill=(255, 0, 0))
                draw.text((10, 35), f"HR: {current_hr:.1f} BPM [{hr_source}]", font=font, fill=(0, 255, 0))
                draw.text((10, 60), time_str, font=font, fill=(255, 255, 255))
                imgtk = ImageTk.PhotoImage(image=img_pil)

                def update_ui(img=imgtk):
                    self.video_label.imgtk = img
                    self.video_label.configure(image=img)

                self.after(0, update_ui)

                data = json.dumps({"emotion": last_emotion, "heart_rate": round(current_hr, 1), "time": time_str})
                broadcast_socket.sendto(data.encode('utf-8'), ('<broadcast>', BROADCAST_PORT))
            except Exception as e:
                print(f"画面渲染或广播异常: {e}")
                continue
        if self.cap is not None:
            self.cap.release()


# ================= 6. 启动程序 =================
if __name__ == "__main__":
    app = BluetoothManagerApp()