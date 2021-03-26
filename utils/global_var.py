import mediapipe as mp


KEY_POINTS_CLASSIFIER_PATH = 'models/lr_21.sav'
DEVICE_ID = 0
WINDOW = "Gesture recognition"

ALPH = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К',
        'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
        'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

classes = ['А', 'Б', 'В', 'Г', 'Е', 'Ж', 'З', 'И', 'Л', 'М',
           'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Х', 'Ш', 'Ы', 'Э']

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
