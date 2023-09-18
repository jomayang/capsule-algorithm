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
        im_i = paths[first_pathological_image_position - iteration_index]
        im_i = colorized_rgb(im_i, model)

      # Colorize image after pathological sequence with iteration_index
      if iteration_index <= len(paths[last_pathological_image_position:-1]): 
        im_j = paths[last_pathological_image_position + iteration_index]
        im_j = colorized_rgb(im_j, model)

  print("[Done] Colorization of all emergency sequence.")


def colorize_rest_of_img(paths, pathological_pointers, len_emergency_seq, model):

    # Colorization of the rest L images.

    if len(pathological_pointers) > 1:

        for y in range(len(pathological_pointers)):
        
            if pathological_pointers[0][0] == pathological_pointers[y][0]: # For first sequence 
        
              next_i = len(paths[:(pathological_pointers[0][0] - len_emergency_seq)])
              next_j = len(paths[(pathological_pointers[0][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])  
              
            elif pathological_pointers[-1][-1] == pathological_pointers[y][-1]: # For last sequences 
        
              next_i = 0
              next_j = len(paths[(pathological_pointers[-1][-1]+len_emergency_seq):-1])
        
            else: # For mid sequences 
        
              next_i = 0
              next_j = len(paths[(pathological_pointers[y][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])
        
            max_ = maximum(next_i, next_j)
        
            for x in range((max_+1)):
        
              if x <= next_i and next_i != 0: # To colorize prior sequence
        
                im_i = paths[(pathological_pointers[y][0]- len_emergency_seq - x)]
                im_i = colorized_rgb(im_i)
                visualize_colorized_image(im_i)
        
              if x <= next_j and next_j != 0: # To colorize next sequence
        
                im_j = paths[(pathological_pointers[y][1] + len_emergency_seq+ x)]
                im_j = colorized_rgb(im_j)
                visualize_colorized_image(im_j)
        print("[Done] Colorization of all grayscale images.")

    else:
        return None

    for y in range(len(pathological_pointers)):
    
        if pathological_pointers[0][0] == pathological_pointers[y][0]:
    
          next_i = len(paths[:(pathological_pointers[0][0] - len_emergency_seq)])
          next_j = len(paths[(pathological_pointers[0][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])  
          
        elif pathological_pointers[-1][-1] == pathological_pointers[y][-1]:
    
          next_i = 0
          next_j = len(paths[(pathological_pointers[-1][-1]+len_emergency_seq):-1])
    
        else: 
    
          next_i = 0
          next_j = len(paths[(pathological_pointers[y][1] + len_emergency_seq):(pathological_pointers[y+1][0] - len_emergency_seq)])
    
        max_ = maximum(next_i, next_j)
    
        for x in range((max_+1)):
    
          if x <= next_i and next_i != 0: 
    
            im_i = paths[(pathological_pointers[y][0]- len_emergency_seq - x)]
            im_i = colorized_rgb(im_i, model)
            #visualize_colorized_image(im_i)
    
          if x <= next_j and next_j != 0: 
    
            im_j = paths[(pathological_pointers[y][1] + len_emergency_seq+ x)]
            im_j = colorized_rgb(im_j, model)
            #visualize_colorized_image(im_j)
    return print("[Done] Colorization of all grayscale images.")