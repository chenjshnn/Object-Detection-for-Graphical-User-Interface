# Xianyu

## Setup

*Tesseract*

```
sudo add-apt-repository -y ppa:alex-p/tesseract-ocr 
sudo apt update
sudo apt install  -y tesseract-ocr
```

*Opencv*

```
python3 -m pip install opencv-python
```

## Test

```
python3 detect.py --test_folder [FOLDER-TO-TEST]
```

For more details, see https://laptrinhx.com/ui2code-how-to-fine-tune-background-and-foreground-analysis-2293652041/