from tkinter import *
from PIL import ImageTk, Image
import os

root = Tk()
root.geometry("600x600")

letters = ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

my_dict = {}

for letter in letters:
    my_dict[letter] = letter + ".jpg"

def show_image():
    letter = letter_input.get().upper()
    img = ImageTk.PhotoImage(Image.open(os.path.join("python/TechVariable/poc2", my_dict[letter])))
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()

letter_input = Entry(root)
letter_input.pack()

button = Button(root, text="Show Image", command=show_image)
button.pack()

root.mainloop()
