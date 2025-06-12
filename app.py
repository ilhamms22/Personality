import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle # Untuk memuat objek preprocessing yang disimpan

app = Flask(__name__)

# --- Muat model yang sudah dilatih sebelumnya dan alat preprocessing ---
# Definisikan path ke file model H5 Anda
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'personality_model.h5')

# Definisikan path ke objek preprocessing Anda yang disimpan
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
LE_GENDER_PATH = os.path.join(os.path.dirname(__file__), 'le_gender.pkl')
LE_EDUCATION_PATH = os.path.join(os.path.dirname(__file__), 'le_education.pkl')
LE_INTEREST_PATH = os.path.join(os.path.dirname(__file__), 'le_interest.pkl')
LE_PERSONALITY_PATH = os.path.join(os.path.dirname(__file__), 'le_personality.pkl')

# Inisialisasi variabel untuk objek yang akan dimuat
model = None
scaler = None
le_gender = None
le_education = None
le_interest = None
le_personality = None

try:
    # Muat model Keras H5
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model Keras H5 berhasil dimuat.")

    # Muat objek preprocessing yang sudah di-fit
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("StandardScaler berhasil dimuat.")

    with open(LE_GENDER_PATH, 'rb') as f:
        le_gender = pickle.load(f)
    print("LabelEncoder untuk Gender berhasil dimuat.")

    with open(LE_EDUCATION_PATH, 'rb') as f:
        le_education = pickle.load(f)
    print("LabelEncoder untuk Education berhasil dimuat.")

    with open(LE_INTEREST_PATH, 'rb') as f:
        le_interest = pickle.load(f)
    print("LabelEncoder untuk Interest berhasil dimuat.")

    with open(LE_PERSONALITY_PATH, 'rb') as f:
        le_personality = pickle.load(f)
    print("LabelEncoder untuk Personality berhasil dimuat.")

except Exception as e:
    print(f"Error saat memuat model atau objek preprocessing: {e}")
    # Jika ada error, pastikan semua objek diatur ke None
    model = scaler = le_gender = le_education = le_interest = le_personality = None


# Pertanyaan Skala 1–5 (sesuai yang Anda berikan)
questions = {
    "Introversion Score": [
        "Saya lebih suka menghabiskan waktu sendiri daripada di tengah keramaian.",
        "Saya merasa lelah setelah bersosialisasi terlalu lama.",
        "Saya lebih nyaman berpikir sebelum berbicara.",
        "Saya lebih suka mendengarkan daripada berbicara di diskusi kelompok.",
        "Saya cenderung mengekspresikan diri secara tertulis daripada verbal."
    ],
    "Sensing Score": [
        "Saya memperhatikan detail kecil dalam situasi sehari-hari.",
        "Saya lebih percaya pada informasi konkret daripada teori abstrak.",
        "Saya menyukai rutinitas dan prosedur yang jelas.",
        "Saya cenderung memercayai pengalaman langsung daripada intuisi.",
        "Saya lebih fokus pada saat ini daripada kemungkinan di masa depan."
    ],
    "Thinking Score": [
        "Saya mengambil keputusan berdasarkan logika, bukan emosi.",
        "Saya lebih suka menyelesaikan konflik dengan alasan rasional.",
        "Saya merasa nyaman mengkritik ide tanpa mengkritik orangnya.",
        "Saya mengutamakan keadilan daripada empati dalam keputusan.",
        "Saya lebih suka kejelasan daripada perasaan yang ambigu."
    ],
    "Judging Score": [
        "Saya menyukai perencanaan dan jadwal yang teratur.",
        "Saya merasa tidak nyaman ketika segala sesuatu tidak pasti.",
        "Saya menyukai tugas yang jelas dan memiliki tenggat waktu.",
        "Saya lebih suka menyelesaikan sesuatu lebih awal daripada mendekati batas waktu.",
        "Saya merasa puas saat bisa mencentang daftar tugas yang selesai."
    ]
}

# --- Data Penjelasan MBTI ---
mbti_explanations = {
    'I': {
        'title': 'Introvert',
        'description': (
            'Kamu cenderung merasa lebih nyaman dan mendapatkan energi dari waktu yang dihabiskan sendirian atau dalam kelompok kecil yang akrab. '
            'Daripada bersosialisasi dalam keramaian, kamu lebih suka melakukan refleksi pribadi, membaca, atau aktivitas yang menenangkan. '
            'Kamu biasanya berpikir terlebih dahulu sebelum berbicara, dan lebih fokus pada dunia dalam (pikiran dan perasaanmu sendiri). '
            'Kehadiranmu mungkin tenang, tetapi pemikiranmu dalam dan penuh makna.'
        )
    },
    'E': {
        'title': 'Ekstrovert',
        'description': (
            'Kamu mendapatkan energi dari lingkungan luar, terutama melalui interaksi sosial, aktivitas kelompok, atau suasana yang ramai. '
            'Kamu cenderung spontan, suka berbicara, dan merasa hidup ketika berhubungan dengan orang lain. '
            'Daripada memproses secara internal, kamu lebih suka berpikir sambil berbicara atau berbagi ide langsung. '
            'Kamu juga cenderung menikmati kegiatan sosial, networking, dan berbagai pengalaman yang melibatkan orang banyak.'
        )
    },
    'S': {
        'title': 'Sensing',
        'description': (
            'Kamu cenderung mengandalkan informasi konkret yang bisa kamu tangkap langsung melalui pancaindera. '
            'Fokusmu lebih ke apa yang nyata dan dapat dibuktikan, seperti fakta, angka, dan pengalaman nyata. '
            'Kamu praktis, detail-oriented, dan suka memecahkan masalah dengan pendekatan yang realistis. '
            'Saat mempelajari sesuatu, kamu lebih suka instruksi yang jelas, langkah demi langkah, daripada teori atau imajinasi abstrak.'
        )
    },
    'N': {
        'title': 'Intuition',
        'description': (
            'Kamu cenderung melihat dunia dengan pendekatan yang imajinatif dan menyeluruh. '
            'Alih-alih berfokus pada detail konkret, kamu lebih tertarik pada pola, konsep, dan kemungkinan-kemungkinan masa depan. '
            'Kamu suka bertanya "bagaimana jika", dan sering melihat koneksi antara hal-hal yang tampak tidak berkaitan. '
            'Gaya berpikirmu cenderung kreatif, inovatif, dan suka mengeksplorasi ide-ide baru yang belum tentu langsung bisa diterapkan.'
        )
    },
    'T': {
        'title': 'Thinking',
        'description': (
            'Ketika harus mengambil keputusan, kamu cenderung menggunakan logika, analisis objektif, dan pertimbangan rasional. '
            'Kamu lebih fokus pada keadilan, efisiensi, dan kebenaran faktual dibanding perasaan pribadi. '
            'Meskipun mungkin terlihat "dingin", kamu sebenarnya hanya ingin adil dan konsisten dalam menilai situasi. '
            'Kamu juga biasanya tidak takut menyampaikan kritik, selama itu membangun dan berdasarkan data atau argumen yang kuat.'
        )
    },
    'F': {
        'title': 'Feeling',
        'description': (
            'Kamu membuat keputusan dengan mempertimbangkan nilai-nilai pribadi dan perasaan orang lain. '
            'Empati, keharmonisan, dan hubungan yang baik sangat penting bagimu. '
            'Kamu cenderung menghindari konflik dan lebih memilih menyelesaikan masalah dengan pendekatan yang lembut dan penuh pengertian. '
            'Ketika orang lain sedih atau tertekan, kamu bisa dengan mudah merasakan dan memahami emosi mereka.'
        )
    },
    'J': {
        'title': 'Judging',
        'description': (
            'Kamu lebih suka menjalani hidup dengan perencanaan, struktur, dan keputusan yang jelas. '
            'Kamu merasa tenang ketika segala sesuatu berjalan sesuai rencana dan jadwal. '
            'Kamu juga cenderung menetapkan tujuan jangka panjang dan bekerja keras secara terorganisir untuk mencapainya. '
            'Ketika menghadapi masalah, kamu biasanya ingin segera menemukan solusi dan menyelesaikannya.'
        )
    },
    'P': {
        'title': 'Perceiving',
        'description': (
            'Kamu menikmati kebebasan dalam menjelajahi berbagai kemungkinan dan tidak terlalu suka dibatasi oleh rencana yang kaku. '
            'Kamu fleksibel, spontan, dan sering menemukan ide atau solusi baru saat berada di tengah proses. '
            'Daripada langsung menyelesaikan tugas, kamu mungkin lebih suka membiarkan opsi tetap terbuka dan menunggu waktu yang tepat. '
            'Gaya hidupmu cenderung mengikuti alur, dan kamu biasanya mudah beradaptasi dengan perubahan yang datang tiba-tiba.'
        )
    }
}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           gender_options=list(le_gender.classes_) if le_gender else ['Lelaki', 'Perempuan'],
                           education_options=list(le_education.classes_) if le_education else ['SD', 'SMP', 'SMA', 'S1'],
                           interest_options=list(le_interest.classes_) if le_interest else ['Arts', 'Others', 'Sports', 'Unknown'],
                           questions=questions)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or le_gender is None or \
       le_education is None or le_interest is None or le_personality is None:
        return render_template('index.html', error_message='Model atau komponen preprocessing tidak dimuat. Mohon periksa log server.',
                               gender_options=['Lelaki', 'Perempuan'],
                               education_options=['SD', 'SMP', 'SMA', 'S1'],
                               interest_options=['Arts', 'Others', 'Sports', 'Unknown'],
                               questions=questions)

    try:
        user_input = request.form

        # Ambil nama (tidak digunakan untuk prediksi model, hanya untuk tampilan jika perlu)
        user_name = user_input.get('name', 'Pengguna')

        age = int(user_input['age'])
        gender_encoded = le_gender.transform([user_input['gender']])[0]
        education_encoded = le_education.transform([user_input['education']])[0]

        # Ambil nilai minat. Jika "Others" dipilih, pastikan kita tetap menggunakan encoded "Others".
        # Nilai dari kotak teks "interest_other" dapat disimpan untuk tujuan logging atau display,
        # tetapi tidak akan diumpankan ke model karena model dilatih dengan kategori diskrit.
        selected_interest = user_input['interest']
        if selected_interest == 'Others':
            # Ambil nilai yang diketikkan di textbox
            other_interest_text = user_input.get('interest_other', 'Tidak Spesifik')
            print(f"Pengguna memilih 'Others' dan mengetik: {other_interest_text}") # Debugging
            # Gunakan nilai encoded untuk 'Others' dari LabelEncoder
            interest_encoded = le_interest.transform(['Others'])[0]
        else:
            interest_encoded = le_interest.transform([selected_interest])[0]


        introversion_score = sum(int(user_input[f'Introversion_Score_{i}']) for i in range(len(questions["Introversion Score"])))
        sensing_score = sum(int(user_input[f'Sensing_Score_{i}']) for i in range(len(questions["Sensing Score"])))
        thinking_score = sum(int(user_input[f'Thinking_Score_{i}']) for i in range(len(questions["Thinking Score"])))
        judging_score = sum(int(user_input[f'Judging_Score_{i}']) for i in range(len(questions["Judging Score"])))

        features = np.array([[
            age,
            gender_encoded,
            education_encoded,
            interest_encoded, # Gunakan encoded interest
            introversion_score,
            sensing_score,
            thinking_score,
            judging_score
        ]])

        numerical_features_to_scale = features[:, [0, 4, 5, 6, 7]].astype(float)
        scaled_numerical_features = scaler.transform(numerical_features_to_scale)

        processed_features = np.copy(features).astype(float)
        processed_features[:, 0] = scaled_numerical_features[:, 0]
        processed_features[:, 4] = scaled_numerical_features[:, 1]
        processed_features[:, 5] = scaled_numerical_features[:, 2]
        processed_features[:, 6] = scaled_numerical_features[:, 3]
        processed_features[:, 7] = scaled_numerical_features[:, 4]

        prediction = model.predict(processed_features)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_personality = le_personality.inverse_transform([predicted_class_index])[0]

        personality_chars = list(predicted_personality)
        explanations_for_result = {}
        for char in personality_chars:
            explanations_for_result[char] = mbti_explanations.get(char, {'title': 'Tidak Diketahui', 'description': 'Penjelasan tidak tersedia.'})

        # Kirim nama pengguna ke halaman hasil
        return render_template('result.html',
                               predicted_personality=predicted_personality,
                               explanations=explanations_for_result,
                               user_name=user_name) # Tambahkan user_name di sini

    except Exception as e:
        print(f"Error dalam prediksi: {e}")
        return render_template('index.html', error_message=f'Terjadi kesalahan saat memproses input Anda: {e}',
                               gender_options=list(le_gender.classes_) if le_gender else ['Lelaki', 'Perempuan'],
                               education_options=list(le_education.classes_) if le_education else ['SD', 'SMP', 'SMA', 'S1'],
                               interest_options=list(le_interest.classes_) if le_interest else ['Arts', 'Others', 'Sports', 'Unknown'],
                               questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
