# Система выявления опасного контента в текстах с использованием методов машинного обучения

Проект классификации текста на базе ruBERT для выявления опасного контента.

## Описание файлов

### `content_analyzer_gui.py`
Файл с кодом реализации GUI интерфейса для классификатора текста на базе ruBERT.

### `learning_ruBert.ipynb`
Файл с кодом реализации обучения модели ruBERT на датасете.

## Ссылки

### Скомпилированная программа и обученная модель
[Google Drive](https://drive.google.com/drive/folders/1jT4IkQW6wZmNo8YS2gTEQ_27kMHZoWQJ?usp=drive_link)

- **Программа**: `/dist/content_analyzer_gui.exe` - скомпилированная программа GUI интерфейса
- **Модель**: `/rubert_toxic_classifier_v3` - обученная модель (эту папку нужно выбирать в GUI программе для работы с текстом)

### Обучение модели
[Google Colab](https://colab.research.google.com/drive/1r5fdTMnGICANVssgRgFIjhg1yjJ8IEcr?usp=sharing) - прямая ссылка на код, с помощью которого модель была обучена на датасете.

## Использование

1. Скачайте скомпилированную программу по пути `/dist/content_analyzer_gui.exe` из Google Drive
2. Скачайте папку с обученной моделью `/rubert_toxic_classifier_v3` из Google Drive
3. Запустите программу `content_analyzer_gui.exe`
4. В GUI интерфейсе выберите папку с моделью `rubert_toxic_classifier_v3`
5. Используйте программу для анализа текста
