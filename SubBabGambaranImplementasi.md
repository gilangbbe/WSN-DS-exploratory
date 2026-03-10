3.6 Gambaran Implementasi
Penelitian ini bersifat analisis, namun untuk memberikan konteks penerapan di lapangan, bagian ini menjelaskan gambaran konseptual bagaimana model Intrusion Detection System (IDS) dari hasil penelitian dapat diimplementasikan pada Wireless Sensor Networks (WSN) di lapangan. Penjelasan ini berfokus pada bagaimana fitur-fitur yang digunakan pada dataset WSN-DS diperoleh dari proses komunikasi aktual antar node, serta bagaimana model machine learning dapat dijalankan secara efisien pada sistem dengan keterbatasan sumber daya.
Dalam sistem WSN, setiap node sensor berperan untuk memantau kondisi lingkungan dan mengirimkan data ke cluster head (CH). Komunikasi antar node diatur menggunakan protokol Carrier Sense Multiple Access – Medium Access Control (CSMA-MAC) yang memastikan node hanya mengirim ketika channel tidak sibuk, sehingga mengurangi collision dan menjaga efisiensi energi. Setelah proses pembentukan cluster, pengiriman data dilakukan berdasarkan jadwal Time Division Multiple Access (TDMA) yang dikendalikan oleh CH. Node dengan urutan rank tertentu hanya boleh mentransmisikan pada slot waktunya. Beberapa fitur seperti Advertise_Sent/Receive, Join_Request, Schedule_Sent/Receive didapatkan pada MAC layer. Sedangkan network layer yang menangani routing antar node mengatur fitur-fitur seperti DATA_Sent/Receive, Distance_to_CH, Send_Code, dan rank. Model IDS hasil penelitian akan menggunakan fitur-fitur ini untuk mendeteksi pola komunikasi abnormal, seperti paket-paket yang dikirim di luar jadwal TDMA dan lonjakan JOIN_Request yang merupakan indikasi adanya serangan.
Secara konseptual sistem IDS menggunakan model hasil penelitian ini dirancang agar sesuai dengan karakteristik protokol LEACH, di mana Cluster Head (CH) dipilih secara acak dan bergantian setiap ronde untuk menyeimbangkan konsumsi energi antar node. Oleh karena itu, model IDS tidak diimplementasikan langsung pada CH, melainkan dijalankan pada sink node atau gateway yang memiliki kapasitas komputasi lebih tinggi dan tidak ikut berganti peran selama operasi jaringan. Secara diagram, arsitektur bisa digambarkan pada Gambar 3.7
Arsitektur tersebut dapat dijelaskan sebagai berikut:
1. Node Sensor
Node sensor berkomunikasi menggunakan protokol CSMA-MAC pada fase pembentukan cluster dan TDMA pada fase transmisi data. Node mengumpulkan informasi komunikasi seperti Advertise_Sent, Join_Request_Sent, Data_Sent, serta estimasi Expanded Energy. Data mentah ini dikirim CH terpilih di ronde tersebut.
2. Cluster Head (CH)
CH berfungsi sebagai pengumpul data dari node di cluster. Karena CH berganti setiap ronde sesuai mekanisme LEACH, perannya sebatas melakukan agregasi sementara terhadap data dan meneruskannya ke sink node atau base station. 
3. Base Station / Sink Node
Base station terkoneksi langsung ke LoRa Gateway dan memiliki dua modul komunikasi, yaitu:
a.	Short-range radio interface misalnya IEEE 802.15.4, ZigBee, atau nRF24L01 untuk menerima data agregasi dari CH 
b.	Long-range radio interface untuk meneruskan data ke gateway.
Perangkat keras yang umum digunakan untuk base station dapat berupa single-board computer seperti Raspberry Pi atau Jetson Nano yang memiliki kapasitas komputasi cukup untuk melakukan pra-pemrosesan dan penyimpanan data sementara.
4. LoRa Gateway
LoRa gateway berfungsi sebagai lokasi utama model IDS dijalankan. Gateway menerima data hasil agregasi dari base station melalui channel LoRa, kemudian menjalankan model machine learning hasil penelitian untuk mendeteksi serangan. Model ini telah dilatih menggunakan teknik oversampling dan diekspor dalam format ringan seperti joblib, ONNX, atau TensorFlow Lite agar dapat di-embed secara efisien. Gateway juga meneruskan hasil deteksi ke control center untuk penyimpanan log dan analisis lanjutan.
5. Control Center
Merupakan sistem backend yang menyimpan catatan aktivitas jaringan dan laporan deteksi serangan.
Untuk mendapatkan data yang diperlukan untuk pelatihan model IDS diperlukan suatu monitoring service yang bekerja dengan biaya serendah mungkin, namun tetap mampu menghasilkan data yang cukup untuk mendeteksi dan mengklasifikasikan berbagai jenis serangan. Dalam arsitektur ini, setiap node sensor turut berpatisipasi dalam proses monitoring, bukan hanya bertindak sebagai pengirim data. Setiap node juga memonitor sekumpulan node di sekitarnya untuk mencatat aktivitasi komunikasi mereka, seperti jumlah paket yang dikirim, diterima, atau hilang selama proses transmisi. Pendekatan ini bertujuan mendistribusikan beban monitoring agar tidak menimbulkan konsumsi energi berlebih pada node tertentu.
Seluruh komunikasi antar node dikendalikan oleh protocol CSMA-MAC, yang memungkinkan node mendeteksi kondisi channel sebelum mengirim paket untuk menghindari collision. Dalam protokol CSMA-MAC, hanya satu node yang dapat mentransmisikan pada satu waktu. Namun, setiap node lain tetap berada dalam mode listening dan dapat melakukan passive overhearing terhadap lalu lintas yang sedang berlangsung. Dengan cara ini, node dapat mendeteksi dan mencatat aktivitas node disekitarnya yang berada dalam jangkauan radio, seperti paket yang dikirim, diterima, atau gagal dikirim. Informasi hasil pemantauan pasif ini kemudian dikumpulkan oleh CH di setiap ronde, sehingga sistem dapat membangun dataset terdistribusi yang mencerminkan perilaku komunikasi jaringan secara menyeluruh. Untuk implementasi di lapangan, metode pengambilan fitur-fitur dapat dijabarkan pada tabel 3.4 sebagai berikut:
Fitur	Metode Pengambilan Fitur
Id	Ditentukan saat inisialisasi jaringan atau berdasarkan alamat MAC.
Time	Waktu terjadinya aktivitas komunikasi, dicatat melalui local clock yang disinkronkan antar node.
Is_CH	Ditentukan oleh algoritma LEACH.
Who_CH	Ditentukan saat node menerima pesan advertisement dari CH.
Distance_To_CH	Estimasi jarak ke CH berdasarkan Received Signal Strength Indicator (RSSI) selama komunikasi CSMA-MAC.
Advertise_Sent / Advertise_Receive	Data ini dapat diperoleh dari buffer komunikasi MAC.
JOIN_Request_Sent / JOIN_Request_Receive	Data ini dapat diperoleh dari buffer komunikasi MAC
Schedule_Sent / Schedule_Receive	Data ini dapat diperoleh dari buffer komunikasi MAC
Rank	Ditentukan oleh protocol TDMA
DATA_Sent / DATA_Receive	Setiap node tidak hanya mencatat aktivitasnya sendiri, tetapi juga aktivitas node tetangga yang dipantau melalui protokol CSMA-MAC dan TDMA
Data_Sent_To_BS	Dihitung dari total paket terkirim pada uplink.
distance_CH_To_BS	Estimasi jarak antara CH dan Base Station, diperoleh dari koordinat atau waktu propagasi sinyal.
send_code	Ditetapkan saat fase advertisement dan digunakan untuk validasi rute data.
Expanded Energy	Didapatkan dari sensor konsumsi energi pada perangkat node.
Melalui mekanisme ini, seluruh node sensor berkontribusi dalam pengumpulan data tanpa menambah overhead signifikan pada satu node tertentu. Data hasil pengamatan dikirim ke CH untuk setiap ronde, lalu diteruskan ke base station dan gateway untuk dilakukan inferensi oleh model IDS. 
