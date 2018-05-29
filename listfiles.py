import glob
files = glob.glob("data/*")
print(files)
for f in files :
    fx = f.split("/")
    fn = fx[1].split(".jpg")
    numbers = list(fn[0])
    print("numbers:",numbers)
