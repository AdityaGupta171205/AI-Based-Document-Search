from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


def export_chat(messages, filename="SmartDoc_Chat.pdf"):

    doc = SimpleDocTemplate(filename)
    elements = []
    styles = getSampleStyleSheet()

    for msg in messages:
        text = f"{msg['role'].upper()}: {msg['content']}"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    return filename
