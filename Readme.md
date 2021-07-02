## Gesture recognition
### Describe
Recognizing gestures of letters of the Russian alphabet with mediapipe hand tracker.
A model from the mediapipe framework is used for palm detection. Gestures classified by landmarks using logistic regression.
To record a word, you need to press "S", record the word using gestures, and press "S" again. After that, using the Levenshtein distance, the closest word to the entered word is output
#### Supported gestures
<table>
  <tr>
    <td style="border: none;"> <img src="utils/readme_img/А.png"  alt="А" width = 120px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Б.png"  alt="Б" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/В.png"  alt="В" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Г.png"  alt="Г" width = 112px height = 112px ></td>
   </tr>
   <tr>
    <td style="border: none;"> <img src="utils/readme_img/Д.png"  alt="Д" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Е.png"  alt="ЕЁ" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Ж.png"  alt="Ж" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/З.png"  alt="З" width = 112px height = 112px ></td>
  </tr>
   <tr>
   <td style="border: none;"> <img src="utils/readme_img/И.png"  alt="ИЙ" width = 112px height = 112px ></td>
   <td style="border: none;"> <img src="utils/readme_img/К.png"  alt="К" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Л.png"  alt="Л" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/М.png"  alt="М" width = 112px height = 112px ></td>
  </tr>
  <tr>
   <td style="border: none;"> <img src="utils/readme_img/Н.png"  alt="Н" width = 112px height = 112px ></td>
   <td style="border: none;"> <img src="utils/readme_img/О.png"  alt="О" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/П.png"  alt="П" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Р.png"  alt="Р" width = 112px height = 112px ></td>
  </tr>
  <tr>
   <td style="border: none;"> <img src="utils/readme_img/С.png"  alt="С" width = 112px height = 112px ></td>
   <td style="border: none;"> <img src="utils/readme_img/Т.png"  alt="Т" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/У.png"  alt="У" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Ф.png"  alt="Ф" width = 112px height = 112px ></td>
  </tr>
  <tr>
   <td style="border: none;"> <img src="utils/readme_img/Х.png"  alt="Х" width = 112px height = 112px ></td>
   <td style="border: none;"> <img src="utils/readme_img/Ц.png"  alt="Ц" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Ч.png"  alt="Ч" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Ш.png"  alt="ШЩ" width = 112px height = 112px ></td>
  </tr>
  <tr>
   <td style="border: none;"> <img src="utils/readme_img/Ы.png"  alt="Ы" width = 112px height = 112px ></td>
   <td style="border: none;"> <img src="utils/readme_img/Ь.png"  alt="ЬЪ" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Э.png"  alt="Э" width = 112px height = 112px ></td>
    <td style="border: none;"> <img src="utils/readme_img/Ю.png"  alt="Ю" width = 112px height = 112px ></td>
  </tr>
  <tr>
   <td style="border: none;"> <img src="utils/readme_img/Я.png"  alt="Я" width = 112px height = 112px ></td>
  </tr>
</table>

#### Example of work
![Alt Text](utils/readme_img/example.gif)
##### Recording word
![Alt Text](utils/readme_img/example_word.gif)

### Instructions for running:

1) Clone this repository:

      `git clone https://github.com/manosh7n/gesture_recognition.git`

      `cd gesture_recognition`
2) Create virtual environment:

     `python -m venv env`

      For linux or macOS:
      `source ./env/bin/activate`

      For Windows:
      `.\env\Scripts\activate.bat`

3) Install the necessary packages:

      `python -m pip install -r requirements.txt`

4) Run App.py:

   `python App.py`

5) To turn it off click *Esc* or *q*

##### Possible problem

If the program shuts down immediately after starting, you can change the `DEVICE_ID` value in `./utils/GlobalVar.py` from 0 to [1, 2, 3, 4]. In my case, the webcam is at number 0.
