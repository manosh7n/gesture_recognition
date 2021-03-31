import mediapipe as mp


DEVICE_ID = 0
WINDOW = "Gesture recognition"

ALPH = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К',
        'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
        'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

# without  Ё, Й, Щ, Ъ
classes = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'К', 'Л', 'М',
           'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
           'Ы', 'Ь', 'Э', 'Ю', 'Я']

KEY_POINTS_CLASSIFIER_PATH = f'models/lr_{len(classes)}.sav'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
