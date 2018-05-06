import cv2
import numpy as np
refPt = []
lx,ly = -15,-15
drawing = False


def click_and_keep(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, lx, ly, drawing

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        refPt = [(x, y)]
        lx = x
        ly = y
        print(x,y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            lx = x
            ly = y
            print(x,y)
        else:
            print "Waiting for input...."

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# Gather our code in a main() function
def getpixels(img, windowName):
    # Read Image
    color = dict()
    color['fg'] = (255, 0, 0)
    color['bg'] = (0, 255, 0)
    h,w = img.shape[0], img.shape[1]
    image = np.zeros((h,w,3))

    # keep looping until the 'q' key is pressed

    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, click_and_keep)
    while True:
        # display the image and wait for a keypress
        # xcor.append(lx)
        # ycor.append(ly)
        image = cv2.circle(image, (lx, ly), 5, color[windowName],5)
        img2 = cv2.circle(img, (lx, ly), 5, color[windowName],5)
        cv2.imshow(windowName, img2)
        key = cv2.waitKey(10) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break
    cv2.destroyWindow(windowName)
    global lx, ly
    lx,ly = -15,-15

    # Close the window will exit the program
    if windowName == 'fg':
        (ycor , xcor) = np.where(image[:,:,0] == 255)
    else:
        (ycor , xcor) = np.where(image[:,:,1] == 255)

    return xcor, ycor





# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
