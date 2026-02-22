# Joshua Rolls Column Simulation

The goal of this project is to simulate how a column would look like if the Joshua Scrolls were wrapped around it like the original intention. 

## Context

The Joshua Rolls are a historic document from the tenth century that showcases the progress of Theodosius in the Roman Sassanid war of the seventh century.
It was originally intended to be a template of the images for a triumphal column. 

That is why the ground upon which the figures move is inclined by about 9% and the figures get bigger as the story progresses, as they would be higher up in a column.

## Steps to Take

### Task 1: Assemble all images. 
Stitch all the high resolution images in the folder jc_images together to have a very long image that represents the circa 10 m by 20 cm original image of the roll.

All the high-resolution images need pre-processing: we need to remove the bottom grey area with the Copyright Biblioteca Apostolica Vaticana and then crop the images so that the white background is removed and only the cream vellum remains.

Prerequisite is that the images should not be scaled up or down to keep the original size increase of the figures as the roll gets longer. 

### Task 2: crop image
remove the text elements from the image to make it ready to be transformed into a sculptural element, approximating how the column would have looked like if the Joshua scrolls were taken as a blueprint.

### Task 3: Add depth to the image. 
Figures and background should all be at a correct depth to indicate what is in the foreground and what in the background. The photo becomes a sculpture.

Create a script that creates a 3D model from an image and its depth map. the script should also be able to change the contrast of the depth map for preprocessing, as well as resizing the depth map to the dimensions of the original file before applying it. the intensity of the depth effect should be configurable. At the end of the creation, show the resulting 3D model with pyglet. Allow for an export that allows the 3D model to be used in Blender.

### Task 4: Create a 3D model
As a basis, we want a 3d model of a column of about 10 to 15 m height and 0.5 m diameter. Overlay the long image with depth on the column, wrapping it around the column at a 9 degree angle so that it snakes upwards

Create python script that creates a 3d model column with applied on it the texture jc_roll_small.jpg  in a way so that it wraps around the column in a helical way and snakes upwards, at an angle of about 9 degrees. meaning it is applied obliquely around the column, with the bottom left corner of the image at the feed of the column and the image wrapping around it at an angle. the top right corner of the image should be the top of the column. calculate the height of the column so that it is appropriate for the entire texture. the result is like a triumphal column, like the trajan column for example. use pyglet to show the finished model. ignore all existing python scripts in the directory, their approach has shown not to work. start from zero.

### Task 5: Add a period correct base to the column
To make it nicer and prepare for 3D printing  

### Task 6: Export the Model
Prep the model to be 3d Printable