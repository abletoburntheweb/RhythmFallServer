# app/note_types.py

class NoteType:
    """Базовый класс для типов нот"""
    KICK = "KickNote"
    SNARE = "SnareNote"
    DEFAULT = "DefaultNote"
    HOLD = "HoldNote"


def create_note(note_type, lane, time, length=None, hold_time=None):
    """Создание словаря ноты для JSON"""
    note = {
        "type": note_type,
        "lane": lane,
        "time": time
    }

    if length is not None:
        note["length"] = length
    if hold_time is not None:
        note["hold_time"] = hold_time

    return note