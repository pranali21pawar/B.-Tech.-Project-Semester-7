# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import dicom
from skimage.filters import threshold_minimum
import numpy as np
import os

inputImage = dicom.read_file('/home/shubham/Practise/image.dcm')
image = inputImage.pixel_array

INPUT_FOLDER = '/home/shubham/Practise/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):    
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


first_patient = load_scan(INPUT_FOLDER + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
plt.show()


thresh = -340
binary =  first_patient_pixels < thresh

#plt.imshow(binary[5], cmap='gray')
#plt.set_title('Result')
#plt.show()


from skimage.morphology import label

myLabels, numLabels = label(binary, neighbors=8, background=0, return_num=True)


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


#segmented_lungs = segment_lung_mask(first_patient_pixels, False)
segmented_lungs_fill = segment_lung_mask(first_patient_pixels, True)

finalStack = segmented_lungs_fill * first_patient_pixels

plt.imshow(first_patient_pixels[100], cmap='gray')

temp_segmented_lungs_fill = segmented_lungs_fill[80]

inv_segmented_lungs_fill = 1- temp_segmented_lungs_fill

plt.figure()
plt.imshow(inv_segmented_lungs_fill, cmap='gray')

#plt.figure()
#plt.hist(finalStack.flatten(), bins=80, color='c')
#plt.xlabel("Hounsfield Units (HU)")
#plt.ylabel("Frequency")
#plt.show()

#plt.figure()
#plt.hist(first_patient_pixels[80].flatten(), bins=80, color='c')

#fig = plt.figure()
tempImage =  finalStack[80]

tempImage400 = tempImage > -400
tempImage500 = tempImage > -500
tempImage600 = tempImage > -600

y = fig.add_subplot(1,3,1)
y.imshow(tempImage400, cmap='gray')

y = fig.add_subplot(1,3,2)
y.imshow(tempImage500 , cmap='gray')

y = fig.add_subplot(1,3,3)
y.imshow(tempImage600, cmap='gray')


#fig = plt.figure()
#
#nodules400 = tempImage400 * tempImage
#nodules500 = tempImage500 * tempImage
#nodules600 = tempImage600 * tempImage
#
#y = fig.add_subplot(1,3,1)
#y.imshow(nodules400, cmap='gray')
#
#y = fig.add_subplot(1,3,2)
#y.imshow(nodules500 , cmap='gray')
#
#y = fig.add_subplot(1,3,3)
#y.imshow(nodules600, cmap='gray')


from skimage.morphology import disk
from skimage.morphology import opening

tempImage400_2 = opening(tempImage400, disk(2))
tempImage400_3 = opening(tempImage400, disk(3))

tempImage500_2 = opening(tempImage500, disk(2))
tempImage500_3 = opening(tempImage500, disk(3))

tempImage600_2 = opening(tempImage600, disk(2))
tempImage600_3 = opening(tempImage602, disk(3))


#fig = plt.figure()

y = fig.add_subplot(2,3,1)
y.imshow(tempImage400_2, cmap='gray')

y = fig.add_subplot(2,3,2)
y.imshow(tempImage400_3, cmap='gray')

y = fig.add_subplot(2,3,3)
y.imshow(tempImage500_2, cmap='gray')

y = fig.add_subplot(2,3,4)
y.imshow(tempImage500_3, cmap='gray')

y = fig.add_subplot(2,3,5)
y.imshow(tempImage600_2, cmap='gray')

y = fig.add_subplot(2,3,6)
y.imshow(tempImage600_3, cmap='gray')


allNodules = tempImage400_2+tempImage400_3+tempImage500_2+tempImage500_3+tempImage600_2+tempImage600_3
allNodules =  allNodules>0


final_image = np.logical_xor(allNodules, inv_segmented_lungs_fill)
plt.figure()
plt.imshow(final_image, cmap = 'gray')


