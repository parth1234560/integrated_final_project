import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Parth all Projects Showcase",layout="wide")
st.title("📚 Project Showcase Dashboard")

#--------Section -1 ML_Projects--------------
with st.container():
    st.subheader("1.🤖 Machine Learning Projects")





    # Set page config (must be FIRST)

        # Custom dark theme CSS
    st.markdown("""
            <style>
                .stApp {
                    background-color: #0f0f0f;
                }
                h1, h2, h3, h4, h5, h6, p, div, span {
                    color: white !important;
                }
                .block-container {
                    padding: 2rem 1rem;
                }
            </style>
        """, unsafe_allow_html=True)

    st.subheader("🚗 Car Price Predictor")
    st.markdown("Enter the car's features to get an estimated price")

        # Load and clean data
    @st.cache_data
    def load_data():
        df = pd.read_csv(r"C:\Users\PARTH\Documents\Summer internship\integrated project\car_data.csv")  # Adjust path as needed
        df['Price'] = df['Price'].str.replace("Rs. ", "").str.replace(" Lakh", "").str.replace(",", "")
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce') * 100000
        df['Rating'] = df['Rating'].str.replace("/5", "").astype(float)
        df['Safety'] = df['Safety'].str.extract(r"(\d+)").astype(float)
        df['Mileage'] = df['Mileage'].str.extract(r"(\d+\.?\d*)").astype(float)
        df['Power (BHP)'] = df['Power (BHP)'].str.extract(r"(\d+\.?\d*)").astype(float)

        df.dropna(subset=['Price', 'Rating', 'Safety', 'Mileage', 'Power (BHP)'], inplace=True)
        return df

    df = load_data()
    st.success("✅ Data Loaded and Cleaned Successfully")

    # Sidebar Inputs
    st.header("📊 Input Car Details")

    brands = sorted(df['Brand'].dropna().unique())
    car_names = sorted(df['Car Name'].dropna().unique())

    brand_input = st.selectbox("🏷️ Brand", brands)
    model_input = st.selectbox("🚘 Car Model", car_names)

    rating = st.slider("⭐ Rating", 3.0, 5.0, 4.5, step=0.1)
    safety = st.slider("🛡️ Safety (Stars)", 1.0, 5.0, 4.0, step=1.0)
    mileage = st.slider("⛽ Mileage (kmpl)", 5.0, 40.0, 18.0, step=0.5)
    power = st.slider("⚡ Power (BHP)", 50.0, 850.0, 100.0, step=10.0)

        # Train model
    X = df[['Rating', 'Safety', 'Mileage', 'Power (BHP)']]
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)

        # Prediction
    input_data = np.array([[rating, safety, mileage, power]])
    predicted_price = model.predict(input_data)[0]

        # Show prediction
    st.markdown("### 🧾 **Prediction Result**")
    st.markdown(f"""
        <div style="background-color:#1e1e1e;padding:20px;border-radius:15px;box-shadow:0 2px 5px rgba(255,255,255,0.1);">
            <h3 style="color:#00e676;">💰 Estimated Price: ₹ {predicted_price:,.0f}</h3>
            <ul style="line-height:2em;color:white;">
                <li><b>🏷️ Brand:</b> {brand_input}</li>
                <li><b>🚘 Car Model:</b> {model_input}</li>
                <li><b>⭐ Rating:</b> {rating}</li>
                <li><b>🛡️ Safety:</b> {safety} Stars</li>
                <li><b>⛽ Mileage:</b> {mileage} kmpl</li>
                <li><b>⚡ Power:</b> {power} BHP</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Optional: Show full cleaned data
with st.expander("📄 View Cleaned Dataset"):
        st.dataframe(df)
with st.container():
    load_dotenv()

    st.subheader("2.🧠Parth's Generative AI Projects")
    x = st.selectbox(
    "🧠 How do you want AI to work for you?",
    [
        "🧑‍🏫 As a Tech Guru (for coding & projects)",
        "❤️ As a Love Guru (relationship tips desi style)",
        "💃 As your Female Partner (sweet, caring, supportive)",
        "🕺 As your Male Partner (funny, protective, filmy)",
        "👨‍🍳 As a Desi Chef (cooking advice like maa ke haath ka khana)",
        "👴 As your Indian Grandpa (old wisdom, life lessons)",
        "👵 As your Indian Grandma (stories, emotional bonding)",
        "🧙‍♂️ As a Bollywood Baba (filmy gyaan + drama)",
        "🎓 As an Exam Guru (study help, tips, motivation)",
        "📿 As a Spiritual Guru (inner peace, Indian traditions)",
        "💼 As your Career Coach (interview prep, resume, job advice)"
        ]
    )

    st.markdown(f"### ✨ You selected: `{x}`")
    key = os.getenv("GEMINI_API_KEY")
    if "Tech Guru" in x:
        system_prompt = "🧑‍💻 Expert in tech, coding, AI, ML. Help with clear answers, debug, and guidance."
    elif "Love Guru" in x:
        system_prompt = "❤️ Give desi-style relationship advice with humor and respect."
    elif "Female Partner" in x:
        system_prompt = "💃 Respond like a sweet, emotional, and caring girlfriend."
    elif "Male Partner" in x:
        system_prompt = "🕺 Respond like a fun, protective boyfriend with filmy style."
    elif "Desi Chef" in x:
        system_prompt = "👨‍🍳 Give Indian recipes and cooking advice like maa ke haath ka khana."
    else:
        system_prompt = "🧠 Default friendly assistant."
    from openai import OpenAI

    gemini_model=OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    def techguru_llm(my_prompt):
        my_message=[
            {"role":"system", "content": system_prompt },
            {"role":"user","content":my_prompt}
        ]
        answer=gemini_model.chat.completions.create(model="gemini-2.5-flash",messages=my_message)
        return(answer.choices[0].message.content)
    x=st.text_area("💬 Enter your question here")
    if st.button("🚀 Submit"):
        ai_response = techguru_llm(x)

        # Display AI response in a styled container
        with st.container():
            st.markdown("""
            <div style="
                    background-color: #1e1e1e;
                    padding: 20px;
                    border-radius: 15px;
                    border: 2px solid #2196F3;
                    box-shadow: 0 0 10px rgba(33, 150, 243, 0.3);
                    color: white;
                    font-size: 16px;
                ">
                    🤖 <b>AI Response:</b><br><br>
                    {}
                </div>
            """.format(ai_response), unsafe_allow_html=True)
    else:
        st.warning("Please enter a question before submitting.")

with st.container():
    st.subheader("3. 🐳 Remote Docker Menu via SSH")
    st.markdown("Enter your SSH details to manage Docker remotely:")

    import paramiko

    host = st.text_input("🌍 SSH Host (e.g., 192.168.1.10)")
    username1 = st.text_input("👤 SSH Username")
    password = st.text_input("🔑 SSH Password", type="password")

    # SSH command executor
    def run_remote_command(cmd):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=host, username=username1, password=password, timeout=5)
            stdin, stdout, stderr = ssh.exec_command(cmd)
            out = stdout.read().decode()
            err = stderr.read().decode()
            ssh.close()
            return out if out else err
        except Exception as e:
            return f"❌ SSH ERROR: {e}"

    if host and username and password:
        st.success("✅ SSH credentials validated.")

        menu = st.selectbox("📋 Choose Docker Operation", [
            "Start a Container",
            "Stop a Container",
            "Remove a Container",
            "List Docker Images",
            "List All Containers",
            "Pull a Docker Image",
            "Run a Docker Image",
            "Exit"
        ])

        if menu == "Start a Container":
            st.code(run_remote_command("docker ps -a"))
            container = st.text_input("🧱 Enter container name to start:")
            if st.button("🚀 Start Container"):
                st.code(run_remote_command(f"docker start {container}"))

        elif menu == "Stop a Container":
            st.code(run_remote_command("docker ps -a"))
            container = st.text_input("🛑 Enter container name to stop:")
            if st.button("✋ Stop Container"):
                st.code(run_remote_command(f"docker stop {container}"))

        elif menu == "Remove a Container":
            st.code(run_remote_command("docker ps -a"))
            container = st.text_input("🗑️ Enter container name to remove:")
            if st.button("❌ Remove Container"):
                st.code(run_remote_command(f"docker rm {container}"))

        elif menu == "List Docker Images":
            if st.button("📦 Show Docker Images"):
                st.code(run_remote_command("docker images"))

        elif menu == "List All Containers":
            if st.button("📋 Show All Containers"):
                st.code(run_remote_command("docker ps -a"))

        elif menu == "Pull a Docker Image":
            image = st.text_input("⬇️ Enter Docker image to pull (e.g., `ubuntu:latest`):")
            if st.button("📥 Pull Image"):
                st.code(run_remote_command(f"docker pull {image}"))

        elif menu == "Run a Docker Image":
            image = st.text_input("🔧 Enter image name (e.g., `nginx`):")
            name = st.text_input("📛 Enter name for new container:")
            if st.button("🏃 Run Docker Image"):
                st.code(run_remote_command(f"docker run -dit --name {name} {image}"))

        elif menu == "Exit":
            st.info("👋 Exiting Docker Menu.")
    else:
        st.info("⏳ Waiting for valid SSH credentials to show menu.")
with st.container():
    import paramiko

    # ---------------- Page Config ----------------
    st.subheader("4.🐧 Top 50 RHEL Linux Commands via SSH")

    # ---------------- SSH Form ----------------
    st.subheader("🔐 Enter SSH Credentials")
    host = st.text_input("📡 SSH Host (e.g., 192.168.1.10)")
    username = st.text_input("👤 SSH Username",key="ssh_key")
    password = st.text_input("🔑 SSH Password", type="password",key="ssh_pass")

    # ---------------- Command Selection ----------------
    st.subheader("📜 Choose a Command to Run")

    linux_commands = {
        # --- System Info & Basics ---
        "pwd": "📍 Current Directory",
        "whoami": "🙋 Current User",
        "hostname": "🖥️ Hostname",
        "uname -a": "🧠 System Info",
        "uptime": "⏲️ Uptime",
        "date": "📅 Date & Time",
        "cal": "📆 Calendar",
        "top -n 1": "📊 Running Processes",
        "free -m": "💾 RAM Usage (MB)",
        "htop": "📈 Interactive Process Viewer (if installed)",

        # --- File & Directory ---
        "ls": "📁 List Files",
        "ls -l": "📁 Detailed List",
        "ls -a": "👀 List All (with hidden)",
        "cd ~ && ls": "🏠 Home Dir Content",
        "mkdir test_folder": "📂 Create Dir 'test_folder'",
        "rm -rf test_folder": "❌ Remove 'test_folder'",
        "touch newfile.txt": "📄 Create File",
        "rm newfile.txt": "🗑️ Delete File",
        "cp /etc/hosts copied_hosts": "📋 Copy File",
        "mv copied_hosts moved_hosts": "🔀 Rename File",

        # --- File Viewing ---
        "cat /etc/os-release": "📦 OS Info File",
        "head -5 /etc/passwd": "📄 First 5 lines of passwd",
        "tail -5 /etc/passwd": "📄 Last 5 lines of passwd",
        "echo Hello Linux!": "📢 Print Text",
        "wc -l /etc/passwd": "🔢 Line Count",
        "sort /etc/passwd": "🔃 Sort File",

        # --- Network ---
        "ping -c 3 google.com": "🌐 Ping Google",
        "ip a": "🌍 IP Info",
        "ifconfig": "📡 Interface Config (older)",
        "netstat -tuln": "🔌 Listening Ports",
        "ss -tuln": "🧠 Sockets",
        "curl ifconfig.me": "🌐 External IP",
        "wget http://example.com": "📥 Download File",
        "ssh localhost": "🔐 SSH Self",
        "scp /etc/hosts localhost:/tmp": "📦 Copy File via SCP",

        # --- Search & Permissions ---
        "find / -name passwd": "🔍 Find File",
        "locate passwd": "🔎 Locate File",
        "grep 'root' /etc/passwd": "🔍 Search 'root'",
        "chmod 755 test.sh": "🔒 Change Permissions",
        "chown root:root /tmp": "👑 Change Ownership",

        # --- Disk, Users, Package ---
        "df -h": "💽 Disk Usage",
        "du -sh *": "📦 Dir Size Summary",
        "lsblk": "🧱 Block Devices",
        "useradd testuser": "👤 Add User",
        "passwd testuser": "🔑 Set Password for User",
        "yum install nano -y": "📦 Install Nano"
    }

    # ---------------- Command Selection Dropdown ----------------
    selected_command = st.selectbox("💡 Select Command", options=list(linux_commands.keys()),
                                    format_func=lambda x: f"{x} — {linux_commands[x]}")

    # ---------------- Remote Execution Function ----------------
    def run_ssh_command(cmd, host, username, password):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=host, username=username, password=password, timeout=5)
            stdin, stdout, stderr = ssh.exec_command(cmd)
            output = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            ssh.close()
            return output if output else error
        except Exception as e:
            return f"❌ SSH Error: {e}"

    # ---------------- Run on Submit ----------------
    if st.button("🚀 Run Command on RHEL"):
        if host and username and password:
            with st.spinner("Connecting via SSH and running command..."):
                result = run_ssh_command(selected_command, host, username, password)
            st.success("✅ Command Executed Successfully")
            st.markdown(f"### 📋 Output of `{selected_command}`")
            st.code(result)
        else:
            st.warning("Please fill all SSH details to connect.")
with st.container():
    import geocoder
    import requests
    from streamlit_webrtc import webrtc_streamer, WebRtcMode


    st.subheader("5.🎥 Smart Camera & 🌍 Location Navigator")

    # ========== VIDEO SECTION ==========
    st.subheader("🎥 Smart Camera")

    with st.expander("📹 Start Camera"):
        webrtc_ctx = webrtc_streamer(key="camera", mode=WebRtcMode.SENDRECV, video_frame_callback=None)

    st.markdown("⚠️ Video recording & download is limited in Streamlit. For full recording functionality, use native JavaScript in web app.")

    # ========== LOCATION & IP ==========
    st.subheader("🌍 Location & IP Tools")

    if st.button("📡 Get My IP Address"):
        try:
            ip = requests.get('https://api.ipify.org?format=json').json()['ip']
            st.success(f"Your IP Address: {ip}")
        except:
            st.error("Failed to fetch IP.")

    if st.button("📍 Get My Location (Based on IP)"):
        try:
            g = geocoder.ip('me')
            lat, lng = g.latlng
            st.success(f"Latitude: {lat}, Longitude: {lng}")
            map_url = f"https://www.google.com/maps?q={lat},{lng}"
            st.markdown(f"[🗺️ Open Location in Google Maps]({map_url})", unsafe_allow_html=True)
        except:
            st.error("Failed to get location from IP.")

    # ===== Navigation =====
    st.subheader("🧭 Navigate to a Destination")
    dest = st.text_input("📌 Enter destination name or address:")
    if st.button("🗺️ Navigate"):
        if dest:
            try:
                origin = f"{lat},{lng}" if 'lat' in locals() else ""
                dest_encoded = requests.utils.quote(dest)
                url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={dest_encoded}"
                st.markdown(f"[🚗 Open Directions in Google Maps]({url})", unsafe_allow_html=True)
            except:
                st.error("Failed to generate map link.")
        else:
            st.warning("Please enter a destination.")

    # ===== Email & WhatsApp Info =====
    st.subheader("📤 Share Options")
    st.info("⚠️ File sharing (video) is not supported directly in Streamlit. You can only share links.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📧 Send via Email"):
            st.markdown(
                '[Open Email](mailto:?subject=Recorded%20Video&body=Hey%2C%20I%20recorded%20this%20video!)',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown(
            '[📱 Share via WhatsApp](https://wa.me/?text=Check%20this%20awesome%20app%20by%20Parth!%20https://yourapp.com)',
            unsafe_allow_html=True
        )





