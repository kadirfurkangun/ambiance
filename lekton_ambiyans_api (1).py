from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic
import json
from typing import List, Optional
import asyncio

app = FastAPI(title="Lekton Ambiance API")

# CORS — Lekton'dan çalışsın
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Claude client
client = Anthropic()

# ─────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────

class TextChunk(BaseModel):
    id: int
    text: str

class AnalyzeRequest(BaseModel):
    book_id: str
    title: str
    author: str
    chunks: List[TextChunk]

class EmotionMode(BaseModel):
    mod: str
    valans: int  # -100 to +100
    uyarılma: int  # -100 to +100
    karanlık: int  # -100 to +100
    ortam: Optional[str] = None
    zaman: Optional[str] = None
    tempo: Optional[str] = None
    renk_ana: str
    renk_vurgu: str
    partikul_tipi: str
    partikul_hizi: int
    partikul_yogunluk: int

class AmbianceBlock(BaseModel):
    chunk_ids: List[int]
    emotion: EmotionMode
    description: str

class AnalyzeResponse(BaseModel):
    book_id: str
    title: str
    author: str
    chunks_analyzed: int
    blocks: List[AmbianceBlock]
    summary: dict

# ─────────────────────────────────────────────────────────
# EMOTION TREE — Tüm modlar
# ─────────────────────────────────────────────────────────

EMOTION_TREE = {
    # ÖFKE
    "ofke_soguk": {
        "valans": -80, "uyarılma": 60, "karanlık": 70,
        "renk_ana": "#2c2c2c", "renk_vurgu": "#8b0000", "renk_nabiz": "#cc0000",
        "ortam": "oda_soguk", "zaman": "gece", "tempo": "orta",
        "partikul_tipi": "duman", "partikul_hizi": 40, "partikul_yogunluk": 60,
        "kat": "ÖFKE", "desc": "Soğuk, bastırılmış öfke. İçsel kaynama, kontrolü kaybetme korkusu."
    },
    "ofke_patlayan": {
        "valans": -95, "uyarılma": 100, "karanlık": 80,
        "renk_ana": "#330000", "renk_vurgu": "#ff0000", "renk_nabiz": "#ff6600",
        "ortam": "savaş_alanı", "zaman": "gece", "tempo": "hizli",
        "partikul_tipi": "kor", "partikul_hizi": 90, "partikul_yogunluk": 100,
        "kat": "ÖFKE", "desc": "Patlayan, kontrol dışı öfke. Şiddet, yıkım, intikam."
    },
    "ofke_intikam": {
        "valans": -85, "uyarılma": 85, "karanlık": 85,
        "renk_ana": "#1a0000", "renk_vurgu": "#cc0000", "renk_nabiz": "#ff3300",
        "ortam": "kapkaracık", "zaman": "gece", "tempo": "yavaş_ve_isitkali",
        "partikul_tipi": "kıvılcım", "partikul_hizi": 50, "partikul_yogunluk": 70,
        "kat": "ÖFKE", "desc": "Öfke + planlama. Intikam, hesaplaşma, adalet arayışı."
    },
    "ofke_hayal_kiriklik": {
        "valans": -70, "uyarılma": 50, "karanlık": 60,
        "renk_ana": "#4a3030", "renk_vurgu": "#8b4513", "renk_nabiz": "#a0522d",
        "ortam": "yıkıntı", "zaman": "gün_batımı", "tempo": "orta",
        "partikul_tipi": "toz", "partikul_hizi": 30, "partikul_yogunluk": 50,
        "kat": "ÖFKE", "desc": "Hayal kırıklığından doğan öfke. Eksik bırakılmışlık, haksızlık."
    },

    # KORKU
    "korku_beklenti": {
        "valans": -60, "uyarılma": 70, "karanlık": 50,
        "renk_ana": "#333366", "renk_vurgu": "#6600cc", "renk_nabiz": "#9933ff",
        "ortam": "hava_kapalı", "zaman": "alacakaranlık", "tempo": "orta",
        "partikul_tipi": "sis", "partikul_hizi": 20, "partikul_yogunluk": 40,
        "kat": "KORKU", "desc": "Bekleyiş, hazırlık. Bilinmeyen ne gelecek? Gerilim artar."
    },
    "korku_panik": {
        "valans": -90, "uyarılma": 100, "karanlık": 90,
        "renk_ana": "#1a001a", "renk_vurgu": "#ff0066", "renk_nabiz": "#ff33cc",
        "ortam": "kapkaracık_labirent", "zaman": "gece", "tempo": "hızlı",
        "partikul_tipi": "ateş", "partikul_hizi": 100, "partikul_yogunluk": 100,
        "kat": "KORKU", "desc": "Panik, kaçış, can sıkıntısı. En yüksek gerilim."
    },
    "korku_varoluşsal": {
        "valans": -75, "uyarılma": 60, "karanlık": 95,
        "renk_ana": "#0a0a0a", "renk_vurgu": "#333333", "renk_nabiz": "#666666",
        "ortam": "uzay_boşluk", "zaman": "gece", "tempo": "donmuş",
        "partikul_tipi": "geometrik_tasılar", "partikul_hizi": 10, "partikul_yogunluk": 20,
        "kat": "KORKU", "desc": "Varoluşsal korku. Ölüm, kaybolma, anlamsızlık."
    },
    "korku_paranoya": {
        "valans": -70, "uyarılma": 75, "karanlık": 75,
        "renk_ana": "#1a1a33", "renk_vurgu": "#663366", "renk_nabiz": "#9966cc",
        "ortam": "yakın_duvarlar", "zaman": "gece", "tempo": "orta_gergin",
        "partikul_tipi": "gözler", "partikul_hizi": 40, "partikul_yogunluk": 60,
        "kat": "KORKU", "desc": "Paranoia, şüphe. İzleniyor mu? Kim güvenilir?"
    },
    "korku_mahcubiyyet": {
        "valans": -50, "uyarılma": 55, "karanlık": 40,
        "renk_ana": "#4d4d4d", "renk_vurgu": "#cc99cc", "renk_nabiz": "#ffccff",
        "ortam": "kalabalık", "zaman": "gün", "tempo": "yavaş",
        "partikul_tipi": "gözler", "partikul_hizi": 20, "partikul_yogunluk": 40,
        "kat": "KORKU", "desc": "Mahcubiyet, utanç. Herkes biliyor mu?"
    },

    # ÜZÜNTÜ
    "uzuntu_melankolik": {
        "valans": -50, "uyarılma": -20, "karanlık": 60,
        "renk_ana": "#36454f", "renk_vurgu": "#708090", "renk_nabiz": "#a9a9a9",
        "ortam": "yağmurlu_bahçe", "zaman": "yağmurlu_gün", "tempo": "yavaş",
        "partikul_tipi": "yağmur", "partikul_hizi": 20, "partikul_yogunluk": 50,
        "kat": "ÜZÜNTÜ", "desc": "Hafif melankolya. Nostalji, geçmiş anılar, eksiklik."
    },
    "uzuntu_kayip": {
        "valans": -70, "uyarılma": -30, "karanlık": 70,
        "renk_ana": "#2f4f4f", "renk_vurgu": "#5f9ea0", "renk_nabiz": "#87ceeb",
        "ortam": "boş_oda", "zaman": "gece", "tempo": "yavaş",
        "partikul_tipi": "yaprak", "partikul_hizi": 10, "partikul_yogunluk": 30,
        "kat": "ÜZÜNTÜ", "desc": "Kayıp, yas, hüzün. Ayrılık, veda, son."
    },
    "uzuntu_depresif": {
        "valans": -85, "uyarılma": -40, "karanlık": 85,
        "renk_ana": "#1a1a1a", "renk_vurgu": "#4a4a4a", "renk_nabiz": "#666666",
        "ortam": "boş_oda_soğuk", "zaman": "gece", "tempo": "yavaş",
        "partikul_tipi": "toz_statik", "partikul_hizi": 5, "partikul_yogunluk": 20,
        "kat": "ÜZÜNTÜ", "desc": "Derin depresyon. Umut yok, enerji yok, boşluk."
    },
    "uzuntu_özsefalet": {
        "valans": -65, "uyarılma": -20, "karanlık": 70,
        "renk_ana": "#3d3d3d", "renk_vurgu": "#696969", "renk_nabiz": "#808080",
        "ortam": "işçi_barınağı", "zaman": "gece", "tempo": "yavaş",
        "partikul_tipi": "tuz_taneleri", "partikul_hizi": 15, "partikul_yogunluk": 40,
        "kat": "ÜZÜNTÜ", "desc": "Öz-şefalet, zavallılık. Kişi kendini yazık görüyor."
    },

    # UTANÇ
    "utanc_sosyal": {
        "valans": -60, "uyarılma": 50, "karanlık": 50,
        "renk_ana": "#663333", "renk_vurgu": "#cc6666", "renk_nabiz": "#ff9999",
        "ortam": "kalabalık", "zaman": "gün", "tempo": "yavaş",
        "partikul_tipi": "gözler", "partikul_hizi": 30, "partikul_yogunluk": 50,
        "kat": "UTANÇ", "desc": "Sosyal utanç. Herkesin bakışı. Açıklanmış olmak."
    },
    "utanc_kendinden_tiksinti": {
        "valans": -75, "uyarılma": 40, "karanlık": 75,
        "renk_ana": "#330033", "renk_vurgu": "#660066", "renk_nabiz": "#990099",
        "ortam": "yatakta", "zaman": "gece", "tempo": "yavaş",
        "partikul_tipi": "asitler", "partikul_hizi": 20, "partikul_yogunluk": 60,
        "kat": "UTANÇ", "desc": "Kendinden tiksinme. Vicdan azabı, pislik hissi."
    },
    "utanc_acizlik": {
        "valans": -65, "uyarılma": 30, "karanlık": 65,
        "renk_ana": "#404040", "renk_vurgu": "#808080", "renk_nabiz": "#a0a0a0",
        "ortam": "zindandaki_hücre", "zaman": "gece", "tempo": "yavaş",
        "partikul_tipi": "ağırlık", "partikul_hizi": 5, "partikul_yogunluk": 40,
        "kat": "UTANÇ", "desc": "Acizlik, güçsüzlük. Yapamıyorum, başaramıyorum."
    },

    # NEFRET
    "nefret_derinlikli": {
        "valans": -95, "uyarılma": 80, "karanlık": 90,
        "renk_ana": "#0a0a0a", "renk_vurgu": "#330000", "renk_nabiz": "#660000",
        "ortam": "boğuk_odun_ateşi", "zaman": "gece", "tempo": "orta_bastırılmış",
        "partikul_tipi": "tütsü", "partikul_hizi": 30, "partikul_yogunluk": 80,
        "kat": "NEFRET", "desc": "Derin nefret. Yok etme isteği, tamamen reddediliş."
    },
    "nefret_hor_gorme": {
        "valans": -80, "uyarılma": 50, "karanlık": 60,
        "renk_ana": "#262626", "renk_vurgu": "#4d4d4d", "renk_nabiz": "#660000",
        "ortam": "tepeden_bakış", "zaman": "gün", "tempo": "yavaş",
        "partikul_tipi": "toz_dalgası", "partikul_hizi": 40, "partikul_yogunluk": 50,
        "kat": "NEFRET", "desc": "Hor görme, aşağılama. Çöpten farksız."
    },

    # TİKSİNTİ
    "tiksinti_fiziksel": {
        "valans": -70, "uyarılma": 60, "karanlık": 50,
        "renk_ana": "#4d3300", "renk_vurgu": "#996633", "renk_nabiz": "#ccaa77",
        "ortam": "çürüyen_yemek", "zaman": "gün", "tempo": "orta",
        "partikul_tipi": "güveler", "partikul_hizi": 50, "partikul_yogunluk": 70,
        "kat": "TİKSİNTİ", "desc": "Fiziksel tiksinti. Kötü koku, kirli, haşere."
    },
    "tiksinti_ahlaki": {
        "valans": -80, "uyarılma": 50, "karanlık": 70,
        "renk_ana": "#330033", "renk_vurgu": "#660066", "renk_nabiz": "#990099",
        "ortam": "çöp_yığını", "zaman": "gece", "tempo": "yavaş",
        "partikul_tipi": "sinekler", "partikul_hizi": 60, "partikul_yogunluk": 80,
        "kat": "TİKSİNTİ", "desc": "Ahlaki tiksinti. İmsallık, hırsızlık, sahtelik."
    },

    # SÜRPRIZ (Olumlu)
    "surpriz_sevindirici": {
        "valans": 60, "uyarılma": 80, "karanlık": -20,
        "renk_ana": "#fffacd", "renk_vurgu": "#ffff00", "renk_nabiz": "#ffa500",
        "ortam": "güneşli_açıklık", "zaman": "gün", "tempo": "hızlı",
        "partikul_tipi": "işıltılı_toz", "partikul_hizi": 60, "partikul_yogunluk": 70,
        "kat": "SÜRPRIZ", "desc": "Sürpriz, sevindirici. Beklenmeyeni oldu, iyi oldu!"
    },
    "surpriz_şok": {
        "valans": 0, "uyarılma": 100, "karanlık": 30,
        "renk_ana": "#ffffff", "renk_vurgu": "#ffff00", "renk_nabiz": "#ff0000",
        "ortam": "flaş", "zaman": "an", "tempo": "duruş",
        "partikul_tipi": "flaş", "partikul_hizi": 200, "partikul_yogunluk": 100,
        "kat": "SÜRPRIZ", "desc": "Tamamen şok. Hiç beklenmeyen, çatlatan gerçek."
    },

    # NEŞEsEVİNCLİK
    "nese_hafif": {
        "valans": 40, "uyarılma": 20, "karanlık": -40,
        "renk_ana": "#fffacd", "renk_vurgu": "#ffffe0", "renk_nabiz": "#ffd700",
        "ortam": "bahçe_hava_hoş", "zaman": "sabah", "tempo": "orta",
        "partikul_tipi": "kelebek", "partikul_hizi": 30, "partikul_yogunluk": 30,
        "kat": "NEŞE", "desc": "Hafif neşe, rahat ruh hali. Basit mutluluk."
    },
    "nese_coşkun": {
        "valans": 85, "uyarılma": 80, "karanlık": -50,
        "renk_ana": "#ffff00", "renk_vurgu": "#ffa500", "renk_nabiz": "#ff6347",
        "ortam": "festival", "zaman": "gün", "tempo": "hızlı",
        "partikul_tipi": "konfeti", "partikul_hizi": 80, "partikul_yogunluk": 100,
        "kat": "NEŞE", "desc": "Coşku, sevinç. Kutlama, zafer, sonsuz mutluluk."
    },
    "nese_aydinlanma": {
        "valans": 70, "uyarılma": 60, "karanlık": -60,
        "renk_ana": "#fffacd", "renk_vurgu": "#ffff99", "renk_nabiz": "#ffff00",
        "ortam": "manastır_şafak", "zaman": "şafak", "tempo": "orta",
        "partikul_tipi": "ışık_sütunları", "partikul_hizi": 20, "partikul_yogunluk": 40,
        "kat": "NEŞE", "desc": "Aydınlanma, epifani. Bilerek bağışlanma, değişim."
    },
    "nese_merhamet": {
        "valans": 50, "uyarılma": 30, "karanlık": -30,
        "renk_ana": "#ffd700", "renk_vurgu": "#ffb6c1", "renk_nabiz": "#ff69b4",
        "ortam": "huzurlu_ev", "zaman": "akşam", "tempo": "yavaş",
        "partikul_tipi": "tütsü", "partikul_hizi": 10, "partikul_yogunluk": 20,
        "kat": "NEŞE", "desc": "Merhamet, şefkat. Başkasını affetmek, anlamak."
    },

    # AŞK / TUTKU
    "ask_gerilim": {
        "valans": 30, "uyarılma": 90, "karanlık": 20,
        "renk_ana": "#ff0033", "renk_vurgu": "#ff6666", "renk_nabiz": "#ff99cc",
        "ortam": "başbaşa_oda", "zaman": "gece", "tempo": "hızlı",
        "partikul_tipi": "kıvılcım", "partikul_hizi": 70, "partikul_yogunluk": 80,
        "kat": "AŞK", "desc": "Aşk gerilimi, tutku. Arzı, yaklaşma, çekişme."
    },
    "ask_sehvet": {
        "valans": 50, "uyarılma": 100, "karanlık": 30,
        "renk_ana": "#ff1493", "renk_vurgu": "#ff69b4", "renk_nabiz": "#ffb6c1",
        "ortam": "rahim_ısısı", "zaman": "gece", "tempo": "hızlı",
        "partikul_tipi": "buhar", "partikul_hizi": 90, "partikul_yogunluk": 100,
        "kat": "AŞK", "desc": "Şehvet, karnal arzı. Bedeni isteme, birleşme."
    },
    "ask_kavusma": {
        "valans": 80, "uyarılma": 50, "karanlık": -40,
        "renk_ana": "#ffb6c1", "renk_vurgu": "#ffc0cb", "renk_nabiz": "#ffdbeb",
        "ortam": "saç_saçayıl", "zaman": "gün", "tempo": "orta",
        "partikul_tipi": "yaprak", "partikul_hizi": 20, "partikul_yogunluk": 40,
        "kat": "AŞK", "desc": "Kavuşma, tamamlanma. Birlikte, güvenli, tamamlandık."
    },
    "ask_zihniyim": {
        "valans": 60, "uyarılma": 40, "karanlık": -30,
        "renk_ana": "#ffff99", "renk_vurgu": "#ffff00", "renk_nabiz": "#ffd700",
        "ortam": "beraber_yürüyüş", "zaman": "gün", "tempo": "yavaş",
        "partikul_tipi": "çiçek", "partikul_hizi": 10, "partikul_yogunluk": 20,
        "kat": "AŞK", "desc": "Zihinsel aşk, ruh çiftliği. Anlamak, tanımak, bağlı olmak."
    },
    "ask_özleme": {
        "valans": -20, "uyarılma": 50, "karanlık": 40,
        "renk_ana": "#b0c4de", "renk_vurgu": "#87ceeb", "renk_nabiz": "#add8e6",
        "ortam": "pencereden_bakmak", "zaman": "akşam", "tempo": "yavaş",
        "partikul_tipi": "yağmur", "partikul_hizi": 15, "partikul_yogunluk": 30,
        "kat": "AŞK", "desc": "Özleme, hasret. Ayrı, ama düşünüyor, bekliyor."
    },
    "ask_imkansiz": {
        "valans": -50, "uyarılma": 60, "karanlık": 70,
        "renk_ana": "#4b0082", "renk_vurgu": "#8b008b", "renk_nabiz": "#9932cc",
        "ortam": "ayrı_pencerelerde", "zaman": "gece", "tempo": "yavaş",
        "partikul_tipi": "canlılar", "partikul_hizi": 20, "partikul_yogunluk": 50,
        "kat": "AŞK", "desc": "İmkansız aşk, yasak. Olması gerekmeyen, acıdan tatlı."
    },

    # BARIŞÇIL/SAKIN
    "barisçil_adalet": {
        "valans": 50, "uyarılma": 30, "karanlık": -20,
        "renk_ana": "#98fb98", "renk_vurgu": "#90ee90", "renk_nabiz": "#00ff00",
        "ortam": "hakim_masa", "zaman": "gün", "tempo": "yavaş",
        "partikul_tipi": "işık", "partikul_hizi": 10, "partikul_yogunluk": 20,
        "kat": "BARIŞÇIL", "desc": "Adalet, doğruluk. Sorumlu tutmak, düzene koymak."
    },
    "barisçil_huzur": {
        "valans": 60, "uyarılma": -10, "karanlık": -30,
        "renk_ana": "#e0ffe0", "renk_vurgu": "#f0fff0", "renk_nabiz": "#90ee90",
        "ortam": "bahçe_sabah", "zaman": "sabah", "tempo": "donmuş",
        "partikul_tipi": "pollen", "partikul_hizi": 5, "partikul_yogunluk": 10,
        "kat": "BARIŞÇIL", "desc": "Huzur, barış. Çatışma yok, her yer sessiz."
    },
    "barisçil_bilgelik": {
        "valans": 55, "uyarılma": 20, "karanlık": 10,
        "renk_ana": "#faf0e6", "renk_vurgu": "#daa520", "renk_nabiz": "#bdb76b",
        "ortam": "kütüphane", "zaman": "öğle", "tempo": "yavaş",
        "partikul_tipi": "toz_kütüphaneden", "partikul_hizi": 10, "partikul_yogunluk": 15,
        "kat": "BARIŞÇIL", "desc": "Bilgelik, anlamışlık. Doğru cevap buldum."
    }
}

# ─────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────

def analyze_chunk(chunk_text: str, book_context: str) -> dict:
    """
    Haiku'ya chunk gönder, duygusal mod çıkar.
    """
    prompt = f"""Seni bir kitap analiz editörü olarak kullanıyorum. 
Verilen metin bölümünü okuyup SADECE duygusal modunu döndür. JSON format.

KİTAP KONTEKSTI: {book_context}

METIN:
{chunk_text[:1500]}

Cevap SADECE JSON:
{{
  "mod": "ofke_soguk",
  "valans": -50,
  "uyarılma": 40,
  "karanlık": 60,
  "ortam": "oda_soguk",
  "zaman": "gece",
  "tempo": "orta",
  "kisa_aciklama": "Soğuk öfke, bastırılmış kaynama"
}}

KURALLAR:
- Modlar: {', '.join(list(EMOTION_TREE.keys())[:20])}... (toplamda 60+ mod)
- valans: -100 (çok olumsuz) to +100 (çok olumlu)
- uyarılma: -100 (sakin) to +100 (uyarılmış)
- karanlık: -100 (parlak) to +100 (karanlık)

JSON SADECE, başka text yok."""

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # JSON extract
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        emotion_data = json.loads(response_text)
        return emotion_data
    
    except json.JSONDecodeError:
        # Hata varsa default dönüş
        return {
            "mod": "uzuntu_melankolik",
            "valans": -20,
            "uyarılma": 0,
            "karanlık": 50,
            "ortam": "neutural",
            "zaman": "gündüz",
            "tempo": "orta",
            "kisa_aciklama": "Parse hatası, varsayılan mod"
        }
    except Exception as e:
        print(f"API Error: {e}")
        return {
            "mod": "uzuntu_melankolik",
            "valans": 0,
            "uyarılma": 0,
            "karanlık": 50,
            "ortam": "neutural",
            "zaman": "gündüz",
            "tempo": "orta",
            "kisa_aciklama": "API hatası"
        }

def build_ambiance_blocks(emotions: list) -> list:
    """
    Ardışık aynı modları bir blokta birleştir.
    """
    if not emotions:
        return []
    
    blocks = []
    current_block = {
        "chunk_ids": [0],
        "mod": emotions[0]["mod"]
    }
    
    for i in range(1, len(emotions)):
        if emotions[i]["mod"] == current_block["mod"]:
            current_block["chunk_ids"].append(i)
        else:
            blocks.append(current_block)
            current_block = {
                "chunk_ids": [i],
                "mod": emotions[i]["mod"]
            }
    
    blocks.append(current_block)
    return blocks

# ─────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "lekton-ambiance-api"}

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_book(request: AnalyzeRequest):
    """
    Kitabı analiz et, duygusal harita çıkar.
    """
    try:
        # Kontekst hazırla
        book_context = f"Kitap: {request.title} by {request.author}"
        
        # Her chunk'ı analiz et
        emotions = []
        for chunk in request.chunks:
            emotion = analyze_chunk(chunk.text, book_context)
            emotions.append(emotion)
            # Rate limiting — 1 saniye bekle
            await asyncio.sleep(0.5)
        
        # Blokları oluştur
        blocks = build_ambiance_blocks(emotions)
        
        # Ambiance block'ları doldur
        ambiance_blocks = []
        for block in blocks:
            mod_key = block["mod"]
            if mod_key in EMOTION_TREE:
                mod_data = EMOTION_TREE[mod_key]
                ambiance_blocks.append(
                    AmbianceBlock(
                        chunk_ids=block["chunk_ids"],
                        emotion=EmotionMode(
                            mod=mod_key,
                            valans=mod_data.get("valans", 0),
                            uyarılma=mod_data.get("uyarılma", 0),
                            karanlık=mod_data.get("karanlık", 0),
                            ortam=mod_data.get("ortam"),
                            zaman=mod_data.get("zaman"),
                            tempo=mod_data.get("tempo"),
                            renk_ana=mod_data.get("renk_ana", "#666666"),
                            renk_vurgu=mod_data.get("renk_vurgu", "#999999"),
                            partikul_tipi=mod_data.get("partikul_tipi", "tuz"),
                            partikul_hizi=mod_data.get("partikul_hizi", 20),
                            partikul_yogunluk=mod_data.get("partikul_yogunluk", 30),
                        ),
                        description=mod_data.get("desc", "")
                    )
                )
        
        summary = {
            "total_chunks": len(request.chunks),
            "total_blocks": len(ambiance_blocks),
            "dominant_emotion": blocks[0]["mod"] if blocks else "unknown",
            "analysis_complete": True
        }
        
        return AnalyzeResponse(
            book_id=request.book_id,
            title=request.title,
            author=request.author,
            chunks_analyzed=len(emotions),
            blocks=ambiance_blocks,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
