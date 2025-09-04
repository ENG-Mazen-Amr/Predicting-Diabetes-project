import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import csv
import customtkinter as ctk


data_path = 'diabetes_data.csv'
model_path = 'diabetes_model.pth'
scaler_path = 'scaler.pkl'

df = pd.read_csv(data_path)
df['Diabetes_binary'] = df['Diabetes_012'].apply(lambda x: 1 if x > 0 else 0)
X = df.drop(['Diabetes_012', 'Diabetes_binary'], axis=1)
y = df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
else:
    scaler = StandardScaler()
    scaler.fit(X_train)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  

class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = DiabetesDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class DiabetesModel(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = DiabetesModel(input_dim=X_train.shape[1])

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100  

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 20 == 0:  # كل 20 epoch اطبع الخسارة
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    model.eval()

# حساب الدقة على اختبار البيانات
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted.squeeze() == torch.tensor(y_test.values)).float().mean()
    print(f"Test Accuracy: {(accuracy.item() )* 100 :.2f}%")

# ----------- واجهة المستخدم -----------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

app = ctk.CTk()
app.title("Diabetes Prediction App")
app.geometry("600x750")
center_window(app, 600, 750)

title_label = ctk.CTkLabel(app, text="Diabetes Prediction Program", font=ctk.CTkFont(size=24, weight="bold"))
title_label.pack(pady=15)

scroll_frame = ctk.CTkScrollableFrame(app, width=580, height=500)
scroll_frame.pack(padx=10, pady=10)

labels = [
    "HighBP (0=No, 1=Yes)",
    "HighChol (0=No, 1=Yes)",
    "CholCheck (0=No, 1=Yes)",
    "BMI (e.g., 28)",
    "Smoker (0=No, 1=Yes)",
    "Stroke (0=No, 1=Yes)",
    "HeartDiseaseorAttack (0=No, 1=Yes)",
    "PhysActivity (0=No, 1=Yes)",
    "Fruits (0=No, 1=Yes)",
    "Veggies (0=No, 1=Yes)",
    "HvyAlcoholConsump (0=No, 1=Yes)",
    "AnyHealthcare (0=No, 1=Yes)",
    "NoDocbcCost (0=No, 1=Yes)",
    "GenHlth (1=Excellent ... 5=Poor)",
    "MentHlth (0-30)",
    "PhysHlth (0-30)",
    "DiffWalk (0=No, 1=Yes)",
    "Sex (0=Female, 1=Male)",
    "Real Age (e.g., 60)",
    "Education (1=Less ... 6=College)",
    "Income (1=Lowest ... 8=Highest)"
]

tooltips = [
    "Do you have high blood pressure? 1=Yes, 0=No",
    "Have you been diagnosed with high cholesterol?",
    "Have you had a cholesterol check recently?",
    "Body Mass Index (BMI), e.g., 28",
    "Are you currently a smoker?",
    "Have you ever had a stroke?",
    "Do you have heart disease or heart attack history?",
    "Do you regularly perform physical activity?",
    "Do you eat fruits regularly?",
    "Do you eat vegetables regularly?",
    "Do you consume heavy amounts of alcohol?",
    "Do you have any healthcare coverage?",
    "Does cost prevent you from seeing a doctor?",
    "Your general health rating from 1=Excellent to 5=Poor",
    "Number of mentally unhealthy days in the last month (0-30)",
    "Number of physically unhealthy days in the last month (0-30)",
    "Do you have difficulty walking or climbing stairs?",
    "Sex: 1=Male, 0=Female",
    "Your real age (will be converted to age group)",
    "Education level from 1 (lowest) to 6 (college)",
    "Income level from 1 (lowest) to 8 (highest)"
]

field_types = [
    "bool", "bool", "bool", "float", "bool", "bool", "bool", "bool", "bool", "bool",
    "bool", "bool", "bool", "int", "int", "int", "bool", "choice", "int", "choice", "choice"
]

choices = {
    "Sex": ["Female", "Male"],
    "Education": ["1", "2", "3", "4", "5", "6"],
    "Income": ["1", "2", "3", "4", "5", "6", "7", "8"]
}

entries = []

def show_help(msg):
    help_window = ctk.CTkToplevel(app)
    w, h = 380, 130
    center_window(help_window, w, h)
    help_window.overrideredirect(True)
    frame = ctk.CTkFrame(help_window)
    frame.pack(expand=True, fill="both", padx=10, pady=10)
    label = ctk.CTkLabel(frame, text=msg, wraplength=320, font=ctk.CTkFont(size=18))
    label.pack(padx=15, pady=15)
    ok_btn = ctk.CTkButton(frame, text="OK", command=help_window.destroy)
    ok_btn.pack(pady=5)
    help_window.grab_set()

for i, label_text in enumerate(labels):
    frame = ctk.CTkFrame(scroll_frame)
    frame.pack(fill="x", pady=3, padx=5)

    label = ctk.CTkLabel(frame, text=label_text, width=260, anchor="w", font=ctk.CTkFont(size=16))
    label.pack(side="left", padx=(10, 5))

    help_btn = ctk.CTkButton(frame, text="❓", width=30, height=28, font=ctk.CTkFont(size=14),
                             command=lambda msg=tooltips[i]: show_help(msg))
    help_btn.pack(side="left")

    field_type = field_types[i]

    if field_type == "bool":
        var = ctk.StringVar(value="0")
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(side="right", padx=10)
        yes_btn = ctk.CTkRadioButton(btn_frame, text="Yes", variable=var, value="1")
        no_btn = ctk.CTkRadioButton(btn_frame, text="No", variable=var, value="0")
        yes_btn.pack(side="left")
        no_btn.pack(side="left")
        entries.append(var)
    elif field_type == "choice":
        name = label_text.split()[0]
        var = ctk.StringVar(value=choices[name][0])
        combo = ctk.CTkOptionMenu(frame, variable=var, values=choices[name])
        combo.pack(side="right", padx=10)
        entries.append(var)
    else:
        entry = ctk.CTkEntry(frame, width=100, font=ctk.CTkFont(size=14))
        entry.pack(side="right", padx=10)
        entries.append(entry)

result_label = ctk.CTkLabel(app, text="", font=ctk.CTkFont(size=18, weight="bold"))
result_label.pack(pady=10)

def convert_age_to_code(age):
    age = int(age)
    if age < 24:
        return 1
    elif age < 35:
        return 2
    elif age < 45:
        return 3
    elif age < 55:
        return 4
    elif age < 65:
        return 5
    else:
        return 6

def predict():
    try:
        vals = []
        for i, entry in enumerate(entries):
            field_type = field_types[i]
            label = labels[i].split()[0]

            if isinstance(entry, ctk.StringVar):
                val = entry.get()
                if field_type == "choice":
                    if label == "Sex":
                        val = 1 if val == "Male" else 0
                    else:
                        val = int(val)
                elif field_type == "bool":
                    val = int(val)
            else:
                val = float(entry.get())
            vals.append(val)

        if len(vals) != 21:
            result_label.configure(text="⚠️ Please fill in all fields", text_color="orange")
            return

        real_age = vals[18]
        vals[18] = convert_age_to_code(real_age)

        # ✅ تعظيم تأثير الخصائص المهمة
        important_features = {
            "HighBP": 1.5,
            "HighChol": 1.5,
            "BMI": 1.3,
            "HeartDiseaseorAttack": 1.4,
            "Stroke": 1.3
        }
        # ⛔️ تقليل تأثير خصائص غير مؤثرة فعلاً على التشخيص
        weak_features = {
          "Income": 0.6,
          "Education": 0.6,
          "CholCheck": 0.7,
          "Sex": 0.7
        }        

        for i, label in enumerate(labels):
            feature_name = label.split()[0]
            if feature_name in important_features:
                vals[i] *= important_features[feature_name]
            elif feature_name in weak_features:
              vals[i] *= weak_features[feature_name]
        # تحويل input إلى DataFrame لتجنب التحذير
        X_input_df = pd.DataFrame([vals], columns=X.columns)
        X_input = scaler.transform(X_input_df)
        X_input_tensor = torch.tensor(X_input, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            pred_prob = model(X_input_tensor).item()

        pred_label = 1 if pred_prob >= 0.5 else 0

        if pred_label == 1:
            result_label.configure(text=f"🔴 High chance of Diabetes ({pred_prob*100:.2f}%)", text_color="red")
        else:
            # ✅ تقليل ثقة النموذج في الحالات السليمة
            adjusted_prob = (1 - pred_prob) * 0.85
            result_label.configure(text=f"🟢 Low chance of Diabetes ({adjusted_prob*100:.2f}%)", text_color="green")

    except Exception as e:
        result_label.configure(text="⚠️ Please enter valid data in all fields", text_color="orange")
        print("Prediction error:", e)


def save_user_data_ctk():
    try:
        # سؤال اسم المستخدم بالنافذة
        def save_and_close():
            username = name_var.get().strip()
            if not username:
                error_label.configure(text="Please enter a name!")
                return

            vals = []
            for i, entry in enumerate(entries):
                field_type = field_types[i]
                label = labels[i].split()[0]

                if isinstance(entry, ctk.StringVar):
                    val = entry.get()
                    if field_type == "choice":
                        if label == "Sex":
                            val = 1 if val == "Male" else 0
                        else:
                            val = int(val)
                    elif field_type == "bool":
                        val = int(val)
                else:
                    val = float(entry.get())
                vals.append(val)

            real_age = vals[18]
            vals[18] = convert_age_to_code(real_age)

            # احفظ البيانات مع اسم المستخدم في ملف csv
            filename = "saved_user_data.csv"
            header = ["Name"] + list(X.columns)
            row = [username] + vals

            file_exists = os.path.isfile(filename)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(row)

            popup.destroy()
            result_label.configure(text=f"✅ Data saved for {username}", text_color="green")

        popup = ctk.CTkToplevel(app)
        popup.geometry("350x150")
        center_window(popup, 350, 150)
        popup.title("Save User Data")

        name_var = ctk.StringVar()
        label = ctk.CTkLabel(popup, text="Enter person name:", font=ctk.CTkFont(size=16))
        label.pack(pady=10)

        entry = ctk.CTkEntry(popup, textvariable=name_var, font=ctk.CTkFont(size=14))
        entry.pack(pady=5)
        entry.focus()

        error_label = ctk.CTkLabel(popup, text="", text_color="red", font=ctk.CTkFont(size=12))
        error_label.pack()

        save_btn = ctk.CTkButton(popup, text="Save", command=save_and_close, width=120, font=ctk.CTkFont(size=14))
        save_btn.pack(pady=10)

        popup.grab_set()

    except Exception as e:
        result_label.configure(text="⚠️ Failed to save data", text_color="orange")
        print("Save error:", e)

def toggle_appearance():
    current_mode = ctk.get_appearance_mode().lower()
    new_mode = "dark" if current_mode == "light" else "light"
    ctk.set_appearance_mode(new_mode)


# زرار تبديل الوضع
appearance_btn = ctk.CTkButton(app, text="Toggle Dark/Light Mode", command=toggle_appearance, width=220, font=ctk.CTkFont(size=14))
appearance_btn.pack(pady=10)

# أزرار التوقع والحفظ
btn_frame = ctk.CTkFrame(app)
btn_frame.pack(pady=15)

predict_btn = ctk.CTkButton(btn_frame, text="Predict Diabetes", command=predict, width=200, font=ctk.CTkFont(size=14, weight="bold"))
predict_btn.pack(side="left", padx=20)

save_btn = ctk.CTkButton(btn_frame, text="Save Data", command=save_user_data_ctk, width=200, font=ctk.CTkFont(size=14, weight="bold"))
save_btn.pack(side="right", padx=20)

app.mainloop()
