import glob
from utils import image_mode_check, maximum, colorized_rgb, visualize_colorized_image
#retrieve pathological sequences pointers indeces.

def retrieve_pathological_seq(path):

    paths = sorted(glob.glob(path + "/*.jpg")) 
    
    sequence_length = 0
    start = 0
    is_moving = False
    
    pathological_positions = []
    pathological_pointers = []
    
    for img_path in paths:
    
      if image_mode_check(img_path) == "RGB":
        
        if is_moving == False:
          start = paths.index(img_path)
          sequence_length = 1
          is_moving = True
        else:
          sequence_length = sequence_length + 1
    
      else:
        if is_moving == True:
          end = (start + (sequence_length - 1)) + 1
          pathological_positions.append((start, sequence_length))
          pathological_pointers.append((start, end))
          
          sequence_length = 0
          start = 0
          is_moving = False
        else:
          continue
    
    print("[Done] retrieve pathological sequences pointers indeces")

    return paths, pathological_pointers, pathological_positions

def colorize_emergency_sequences(paths, pathological_pointers, emergency_sequence_length, model):
  
  if len(pathological_pointers) == 1:
    print("One pathological sequence has been detected...")
  else:
    print(f"{len(pathological_pointers)} pathological sequences has been detected...")

  number_of_iterations = maximum(
       len(paths[:pathological_pointers[0][0]]), 
       len(paths[pathological_pointers[0][1]:-1])
       ) if len(pathological_pointers) == 1 else emergency_sequence_length
  
  for sequence in pathological_pointers:  
    
    print(f"Colorizing the {sequence} is on progress..")

    for iteration_index in range(number_of_iterations):

      first_pathological_image_position = sequence[0]
      last_pathological_image_position = sequence[1]

      # Colorize image before pathological sequence with iteration_index
      if iteration_index <= len(paths[:first_pathological_image_position]): 
        image_to_colorize_before = paths[first_pathological_image_position - iteration_index]
        colorized_image_before = colorized_rgb(image_to_colorize_before, model)
        
        # visualize_colorized_image(colorized_image_before)
        
      # Colorize image after pathological sequence with iteration_index
      if iteration_index <= len(paths[last_pathological_image_position:-1]): 
        image_to_colorize_after = paths[last_pathological_image_position + iteration_index]
        colorized_image_after = colorized_rgb(image_to_colorize_after, model)
        
        # visualize_colorized_image(colorized_image_after)
      

  print("[Done] Colorization of all emergency sequence.")
  

def colorize_remaining_images(paths, pathological_pointers, emergency_sequence_length, model):

    # Colorization of the rest L images.
    if len(pathological_pointers) < 1:
      print('No pathological sequences found!')
      return
    

    for iteration_index in range(len(pathological_pointers)):
    
        if pathological_pointers[0][0] == pathological_pointers[iteration_index][0]: # For first sequence 
          
          next_image_before_sequence = len(paths[
              :(pathological_pointers[0][0] - emergency_sequence_length)
            ])
          next_image_after_sequence = len(paths[
              (pathological_pointers[0][1] + emergency_sequence_length)
              :
              (pathological_pointers[iteration_index + 1][0] - emergency_sequence_length)
            ])  
          
        elif pathological_pointers[-1][-1] == pathological_pointers[iteration_index][-1]: # For last sequences 
    
          next_image_before_sequence = 0
          next_image_after_sequence = len(paths[
                (pathological_pointers[-1][-1] + emergency_sequence_length)
                :-1
              ])
    
        else: # For mid sequences 
    
          next_image_before_sequence = 0
          next_image_after_sequence = len(paths[
                (pathological_pointers[iteration_index][1] + emergency_sequence_length)
                :
                (pathological_pointers[iteration_index + 1][0] - emergency_sequence_length)
              ])
    
        number_of_iterations = maximum(next_image_before_sequence, next_image_after_sequence)
    
        for x in range(number_of_iterations + 1):
    
          if x <= next_image_before_sequence and next_image_before_sequence != 0: # To colorize prior sequence
    
            image_to_colorize_before = paths[
                pathological_pointers[iteration_index][0]
                - emergency_sequence_length 
                - x ]
            
            colorized_image_after = colorized_rgb(image_to_colorize_before)
            visualize_colorized_image(colorized_image_after)
    
          if x <= next_image_after_sequence and next_image_after_sequence != 0: # To colorize next sequence
    
            image_to_colorize_after = paths[
                pathological_pointers[iteration_index][1] 
                + emergency_sequence_length 
                + x ]
            
            colorized_image_after = colorized_rgb(image_to_colorize_after)
            visualize_colorized_image(colorized_image_after)

    print("[Done] Colorization of all grayscale images.")

    
