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

### Task 4: Create a 3D model
As a basis, we want a 3d model of a column of about 10 to 15 m height and 0.5 m diameter. Overlay the long image with depth on the column, wrapping it around the column at a 9 degree angle so that it snakes upwards

Create python script that creates a 3d model column. Apply on it the texture jc_roll.jpg  in a way so that it wraps around the column in a helical way and snakes upwards, at an angle of about 9 degrees. calculate the height of the column so that it is appropriate for the entire texture. the output should be glb. ignore all existing python scripts in the directory.

### Task 5: Add a period correct base to the column
To make it nicer and prepare for 3D printing  

### Task 6: Export the Model
Prep the model to be 3d Printable