from PIL import Image
import os.path
import glob
import cv2
def convertjpg(jpgfile,outdir,width=64,height=64):
    # img = cv2.imread(jpgfile)
    img=Image.open(jpgfile)
    try:
        # new_img = cv2.resize(img, (width, height))
        # cv2.imwrite(os.path.join(outdir,os.path.basename(jpgfile)), new_img)
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("F:\\Practice\\test11\\*.jpg"):
    convertjpg(jpgfile,r"F:\Practice\test_resize64")

# im = imread(root_path)
# im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)