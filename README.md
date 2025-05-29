# 🤖 Gesture-Based Communication Assistant

This project is a **Gesture Recognition Communication Assistant** designed to interpret hand gestures in real time using a webcam. It helps individuals, especially those with speech or hearing impairments, to communicate using hand signs. The system leverages **MediaPipe**, **OpenCV**, **PyTTSx3**, and gesture analysis to detect hand gestures and provide real-time audio and visual feedback.

---

## 📸 Features

- 🎯 **Real-Time Hand Gesture Recognition**
- 🔊 **Text-to-Speech Output in Multiple Languages** (English, Spanish, French)
- 🧠 **Customizable Gesture-to-Message Mapping**
- 📁 **Logs for Communication and Errors**
- 🌐 **Dynamic Language Switching Using Gestures**
- 📷 **Live Camera Feed with Gesture Overlays and FPS Counter**

---

## ✋ Supported Gestures and Meanings

| Gesture        | Meaning        |
|----------------|----------------|
| 👍 thumbs_up   | Yes            |
| 👎 thumbs_down | No             |
| ✊ fist         | Stop           |
| 🖐 palm_together| Hello         |
| ☝ one          | One            |
| ✌ two          | Two / Switch Language |
| 🤟 iloveyou     | I love you     |
| 👌 ok           | Okay           |
| 🤙 shaka        | Call me        |
| 🖐✌✌✌ four fingers | Four         |
| 🖐✌✌✌✌ five fingers | Palm        |

---

## 🧩 Technologies Used

- [Python 3.x](https://www.python.org/)
- [MediaPipe](https://google.github.io/mediapipe/) – for hand landmark detection
- [OpenCV](https://opencv.org/) – for image processing and camera feed
- [PyTTSx3](https://pyttsx3.readthedocs.io/) – for offline text-to-speech
- [NumPy](https://numpy.org/) – for numerical analysis
- [JSON](https://www.json.org/) – for configuration

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/shiboshreeroy/gesture-communication-assistant.git
cd gesture-communication-assistant
````

### 2. Install dependencies

```bash
pip install opencv-python mediapipe pyttsx3 numpy
```

> **Note:** On Linux systems, you may also need `espeak` or `libespeak1` for text-to-speech to work.

### 3. Run the application

```bash
python main.py
```

---

## 🗂 Project Structure

```
gesture-communication-assistant/
│
├── main.py                   # Entry point
├── gestures.json             # Custom gesture-message mappings
├── error_log.txt             # Runtime error logs
├── communication_log.txt     # Timestamped communication logs
└── README.md                 # Project overview
```

---

## 🛠 Customization

You can customize gestures and their associated messages in the `gestures.json` file:

```json
{
    "thumbs_up": "Yes",
    "thumbs_down": "No",
    "two": "Change Language"
}
```

---

## 🚀 Future Improvements

* Add GUI support with Tkinter or PyQt
* Add gesture training and user-specific calibration
* Export communication logs to PDF
* Add voice command fallback

---

## 👨‍💻 Author

**Shiboshree Roy**
Full Stack Developer & Open Source Contributor
[GitHub](https://github.com/shiboshree-roy) • [LinkedIn](https://www.linkedin.com/in/shiboshree-roy)

---

## 📜 License

This project is open source under the [MIT License](LICENSE).


