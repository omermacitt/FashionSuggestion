# 👗 Fashion AI - Akıllı Moda Önerileri

Yapay zeka teknolojisi ile moda önerileri sunan modern web uygulaması. Kullanıcıların yükledikleri kıyafet fotoğraflarını analiz ederek benzer ürünler önerir.

![Fashion AI Demo](https://img.shields.io/badge/Demo-Live-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.0+-red) ![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)

## 🚀 Özellikler

### 🔍 AI Teknolojileri
- **YOLO Nesne Tespiti**: Kıyafet türlerini otomatik tespit eder
- **CLIP Görsel Analizi**: Görselleri semantik vektörlere dönüştürür
- **Qdrant Vektör Arama**: Milyonlarca ürün arasında benzerlik araması
- **MinIO Object Storage**: Yüksek performanslı görsel depolama

### 🎨 Modern UI/UX
- **Glass Morphism Tasarım**: Modern cam efektli arayüz
- **Responsive Layout**: Mobil ve masaüstü uyumlu
- **Drag & Drop Upload**: Sürükle-bırak dosya yükleme
- **Real-time Progress**: Canlı ilerleme takibi
- **Interactive Animations**: Yumuşak geçiş animasyonları

### 🔒 Güvenlik
- **Environment Variables**: Güvenli yapılandırma yönetimi
- **No Hardcoded Credentials**: Kimlik bilgileri kodda gömülü değil
- **Docker Containerization**: İzole çalışma ortamı
- **API Rate Limiting**: Kötüye kullanım koruması

## 🛠️ Teknoloji Yığını

### Backend
- **Flask**: Web framework
- **OpenCV**: Görüntü işleme
- **PyTorch**: Deep learning framework
- **Ultralytics YOLO**: Nesne tespit modeli
- **OpenAI CLIP**: Görsel-metin embedding modeli
- **Qdrant**: Vektör veritabanı
- **MinIO**: S3-compatible object storage
- **boto3**: AWS SDK for Python

### Veri Toplama & Etiketleme
- **Selenium**: Web scraping için otomatik görsel toplama
- **Labelme**: Görsel etiketleme ve anotasyon aracı
- **Custom Scripts**: Veri temizleme ve preprocessing

### Frontend
- **HTML5/CSS3**: Modern web standartları
- **Bootstrap 5**: Responsive UI framework
- **JavaScript ES6+**: Modern JavaScript
- **Google Fonts**: Tipografi (Inter)
- **Bootstrap Icons**: İkon seti

### DevOps & Tools
- **Docker**: Konteynerleştirme
- **Docker Compose**: Multi-container orchestration
- **Git**: Versiyon kontrolü
- **Environment Variables**: Yapılandırma yönetimi

## 📋 Sistem Gereksinimleri

### Minimum Gereksinimler
- **Python**: 3.8 veya üzeri
- **RAM**: 8GB (AI modelleri için)
- **Disk**: 5GB boş alan
- **Docker**: 20.10 veya üzeri
- **Docker Compose**: 2.0 veya üzeri

### Önerilen Gereksinimler
- **Python**: 3.11
- **RAM**: 16GB
- **GPU**: CUDA destekli (opsiyonel, hızlandırma için)
- **Disk**: 10GB+ SSD

## 🚀 Kurulum

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/yourusername/FashionSuggestion.git
cd FashionSuggestion
```

### 2. Environment Dosyasını Oluşturun
```bash
cp .env.example .env
```

`.env` dosyasını düzenleyin:
```env
# MinIO Configuration
MINIO_ROOT_USER=your_admin_user
MINIO_ROOT_PASSWORD=your_secure_password
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

# S3 Configuration for Backend
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_BUCKET_NAME=fashion-uploads

# Qdrant Configuration
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
```

### 3. Docker ile Çalıştırın
```bash
# Servisleri başlat
docker-compose up -d

# Logları takip et
docker-compose logs -f
```

### 4. Manuel Kurulum (Alternatif)

#### Backend Kurulumu
```bash
cd backend
pip install -r requirements.txt

# AI modellerini indir
python -c "
from app.models.yolo_detector import YOLODetector
from app.models.clip_embedder import CLIPEmbedder
YOLODetector()  # YOLO modelini indir
CLIPEmbedder()  # CLIP modelini indir
"

# Flask uygulamasını başlat
python -m app.main
```

#### Frontend Kurulumu
```bash
cd frontend
pip install -r requirements.txt

# Template server'ı başlat
python app.py
```

## 🎯 Kullanım

### 1. Web Arayüzü
1. Browser'da `http://localhost:5000` adresine gidin
2. Kıyafet fotoğrafınızı yükleyin (JPG, PNG, WEBP)
3. "Analizi Başlat" butonuna tıklayın
4. AI analizi tamamlandığında benzer ürünleri görün

### 2. API Endpoints

#### Dosya Yükleme
```bash
curl -X POST http://localhost:3000/api/load-file \
  -F "file=@your-image.jpg"
```

#### Nesne Tespiti
```bash
curl -X POST http://localhost:3000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"file_id": "your-file-id"}'
```

#### Benzer Ürün Arama
```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"file_id": "your-file-id"}'
```

#### Sonuçları Görüntüleme
```bash
curl http://localhost:3000/api/result/your-file-id
```

## 🏗️ Proje Yapısı

```
FashionSuggestion/
├── backend/                    # Backend Flask uygulaması
│   ├── app/
│   │   ├── models/            # AI modelleri (YOLO, CLIP)
│   │   ├── services/          # İş mantığı servisleri
│   │   ├── api/               # API endpoint'leri
│   │   └── utils/             # Yardımcı fonksiyonlar
│   └── requirements.txt       # Python bağımlılıkları
├── frontend/                  # Frontend web arayüzü
│   ├── templates/             # HTML şablonları
│   ├── static/
│   │   ├── css/              # Stil dosyaları
│   │   └── js/               # JavaScript dosyaları
│   └── requirements.txt
├── docker-compose.yml         # Multi-container setup
├── .env                       # Environment variables
├── .gitignore                # Git ignore kuralları
└── README.md                 # Bu dosya
```

## 🔧 Yapılandırma

### MinIO Ayarları
- **Console**: http://localhost:9001
- **API**: http://localhost:9000
- **Buckets**: `user-uploads`, `training-data`, `cropped-objects`

### Qdrant Ayarları
- **REST API**: http://localhost:6333
- **gRPC**: localhost:6334
- **Dashboard**: http://localhost:6333/dashboard

### AI Model Ayarları
- **YOLO Model**: YOLOv8n (hızlı tespit için)
- **CLIP Model**: ViT-B/32 (dengeli performans)
- **Embedding Boyutu**: 512 (CLIP default)

## 🧪 Test Etme

### Unit Testler
```bash
cd backend
python -m pytest tests/
```

### API Testleri
```bash
# Postman collection kullanın veya curl ile test edin
curl -X GET http://localhost:3000/api/health
```

### Load Testing
```bash
# Apache Bench ile
ab -n 100 -c 10 http://localhost:3000/api/health
```

## 📊 Performans Metrikleri

### Tespit Performansı
- **YOLO Inference**: ~50ms/görsel
- **CLIP Embedding**: ~100ms/görsel
- **Qdrant Search**: ~10ms/sorgu
- **End-to-end**: ~200ms/istek

### Desteklenen Formatlar
- **Görsel**: JPG, PNG, WEBP, GIF, BMP
- **Maksimum Boyut**: 16MB
- **Minimum Çözünürlük**: 224x224px
- **Önerilen Çözünürlük**: 512x512px+

### 📊 Veri Seti Bilgileri
- **Veri Toplama**: Selenium ile web scraping yapılarak e-ticaret sitelerinden moda görselleri toplanmıştır
- **Etiketleme**: Labelme aracı kullanılarak manuel olarak kıyafet kategorileri etiketlenmiştir
- **Mevcut Kategori Sayısı**: Başlangıç aşamasında temel kategoriler (ceket, gömlek, pantolon vb.)
- **Gelecek Geliştirmeler**: Daha detaylı kategoriler, renk etiketleri, ve çeşitli moda tarzları eklenecektir
- **Veri Genişletme**: Projenin geliştirme aşamasında daha geniş bir ürün yelpazesi dahil edilecektir

## 🚨 Sorun Giderme

### Yaygın Sorunlar

#### 1. Docker Container'ları Başlamıyor
```bash
# Port çakışması kontrolü
netstat -tulpn | grep :9000
netstat -tulpn | grep :6333

# Container'ları yeniden başlat
docker-compose down
docker-compose up -d
```

#### 2. AI Modelleri İndirilemiyor
```bash
# Internet bağlantısını kontrol edin
# HuggingFace cache'i temizleyin
rm -rf ~/.cache/huggingface/
```

#### 3. MinIO Bağlantı Hatası
```bash
# MinIO container'ı kontrol edin
docker-compose logs minio

# Environment variables'ları kontrol edin
echo $MINIO_ROOT_USER
echo $MINIO_ROOT_PASSWORD
```

#### 4. Qdrant Vektör Arama Hatası
```bash
# Qdrant collection'ları kontrol edin
curl http://localhost:6333/collections

# Collection oluşturun
curl -X PUT http://localhost:6333/collections/fashion \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 512, "distance": "Cosine"}}'
```

### Log İnceleme
```bash
# Tüm servis logları
docker-compose logs

# Specific servis
docker-compose logs backend
docker-compose logs qdrant
docker-compose logs minio
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

### Geliştirme Ortamı
```bash
# Development mode
export FLASK_ENV=development
export FLASK_DEBUG=1
```
## 🙏 Teşekkürler

- **Ultralytics**: YOLO modeli için
- **OpenAI**: CLIP modeli için
- **Qdrant**: Vektör arama teknolojisi için
- **MinIO**: Object storage çözümü için
- **Bootstrap**: UI framework için
- **Selenium**: Web scraping altyapısı için
- **Labelme**: Görsel etiketleme aracı için

---

⭐ Bu projeyi beğendiyseniz, lütfen GitHub'da yıldız verin!

**Not**: Bu proje eğitim amaçlı geliştirilmiştir. Production kullanımı için ek güvenlik ve optimizasyonlar gerekebilir.