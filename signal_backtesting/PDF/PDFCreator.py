import copy
import reportlab.rl_config
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import fonts,colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, PageBreak, Table, TableStyle, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import platform
import os



reportlab.rl_config.warnOnMissingFontGlyphs = 0
# pdfmetrics.registerFont(TTFont('song', 'SURSONG.TTF'))    # lidh-221109

# Define font paths for different operating systems
FONT_PATHS = {
    'Windows': [
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
    ],
    'Darwin': [  # macOS
        '/System/Library/Fonts/PingFang.ttf',
        '/System/Library/Fonts/STHeiti Light.ttf',
        '/System/Library/Fonts/STHeiti Medium.ttf',
        '/Library/Fonts/Arial Unicode.ttf'
    ]
}

def find_available_font():
    system = platform.system()
    if system in FONT_PATHS:
        for font_path in FONT_PATHS[system]:
            if os.path.exists(font_path):
                return font_path
    return None

# Try to find and register a suitable font
font_path = find_available_font()
if font_path:
    try:
        pdfmetrics.registerFont(TTFont('heiti', font_path))
    except:
        print("Warning: Failed to register Chinese font. Using default font.")
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        from reportlab.pdfbase._fontdata import FF_ROMAN
        pdfmetrics.registerFont(TTFont('heiti', 'Helvetica'))
else:
    print("Warning: No suitable Chinese fonts found. Using default font.")
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    from reportlab.pdfbase._fontdata import FF_ROMAN
    pdfmetrics.registerFont(TTFont('heiti', 'Helvetica'))

# fonts.addMapping('song', 0, 0, 'song')    # lidh-221110
# fonts.addMapping('song', 0, 1, 'song')    # lidh-221110
fonts.addMapping('heiti', 0, 0, 'heiti')        # lidh-221110
fonts.addMapping('heiti', 0, 1, 'heiti')        # lidh-221110
fonts.addMapping('heiti', 1, 0, 'heiti')
fonts.addMapping('heiti', 1, 1, 'heiti')
stylesheet = getSampleStyleSheet()
titleStyle = copy.deepcopy(stylesheet['title'])
titleStyle.fontName ='heiti'
h1Style = copy.deepcopy(stylesheet['h1'])
h1Style.fontName ='heiti'
h2Style = copy.deepcopy(stylesheet['h2'])
h2Style.fontName ='heiti'
normalStyle = copy.deepcopy(stylesheet['Normal'])
normalStyle.fontName ='heiti'


def convert_imgs_to_pdf(pdf_filename, img_files, space=None):
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
    parts = []
    for img_file in img_files:
        nw, nh = _resize_img_for_a4(img_file)
        parts.append(Image(img_file, width=nw * inch, height=nh * inch))
        if space:
            parts.append(Spacer(1, space * inch))
    doc.build(parts)

A4_WIDTH=8.3

def _resize_img_for_a4(img_filename):
    from PIL import Image
    im = Image.open(img_filename)
    ims = im.size
    nw = A4_WIDTH * 0.69
    nh = nw / (ims[0] / ims[1])
    return nw, nh


class PDFCreator:
    def __init__(self, filename):
        self._filename = filename
        self._line_space_n = 0.05
        self._story = []

    def _spacer(self, n=None):
        if not n:
            n = self._line_space_n
        return Spacer(1, n * inch)

    def build(self):
        doc = SimpleDocTemplate(self._filename, pagesize=A4)
        doc.build(self._story)

    def table(self, data, col_widths=None, highlight_first_row=True):
        if col_widths:
            col_widths = col_widths*inch
        else:
            col_widths = A4_WIDTH*0.69 / len(data[0]) * inch
        if highlight_first_row:
            r1 = []
            for d in data[0]:
                r1.append(Paragraph('<b>{}</b>'.format(d), style=normalStyle))
            data = [r1] + data[1:]
        t = Table(data, colWidths=col_widths)
        # t.setStyle(TableStyle([('FONT', (0,0), (-1,-1), 'song'), ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black), ('BOX', (0,0), (-1,-1), 0.25, colors.black)]))
        t.setStyle(TableStyle([('FONT', (0,0), (-1,-1), 'heiti'), ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black), ('BOX', (0,0), (-1,-1), 0.25, colors.black)]))   # lidh-221110
        self._story.append(self._spacer(0.1))
        self._story.append(t)
        self._story.append(self._spacer(0.2))

    def title(self, text):
        self._story.append(Paragraph(text, titleStyle))
        self._story.append(self._spacer(0.2))

    def h1(self, text):
        self._story.append(self._spacer(0.2))
        self._story.append(Paragraph(text, h1Style))

    def h2(self, text):
        self._story.append(Paragraph(text, h2Style))

    def text(self, text, style=normalStyle):
        if type(text) is list:
            for t in text:
                self._story.append(Paragraph(t, style))
        else:
            self._story.append(Paragraph(text, style))

    def blank_line(self):
        self._story.append(self._spacer(0.1))

    def image(self, img_filename):
        nw, nh = _resize_img_for_a4(img_filename)
        im = Image(img_filename, width=nw*inch, height=nh*inch)
        self._story.append(self._spacer())
        self._story.append(im)
        self._story.append(self._spacer())
