import os

# Từ điển ánh xạ các lớp vi phạm an toàn lao động
VIOLATION_CLASSES = {
    'no-helmet': {'en': 'No Helmet', 'emoji': '⛑️'},
    'no-safety-glasses': {'en': 'No Safety Glasses', 'emoji': '👓'},
    'no-safety-vest': {'en': 'No Safety Vest', 'emoji': '🦺'},
    'phone-usage': {'en': 'Phone Usage', 'emoji': '📱'},
    'using-mobile-beside-construction-machine': {'en': 'Phone use near construction machine', 'emoji': '⚠️'},
}

# Cấu hình Font chữ hệ thống
FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

EMOJI_FONT_PATHS = [
    "C:/Windows/Fonts/seguiemj.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc"
]

def get_font_paths():
    text_font = next((p for p in FONT_PATHS if os.path.exists(p)), "arial.ttf")
    emoji_font = next((p for p in EMOJI_FONT_PATHS if os.path.exists(p)), text_font)
    return text_font, emoji_font