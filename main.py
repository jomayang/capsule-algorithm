import glob
from utils import image_mode_check, maximum, colorized_rgb, visualize_colorized_image
#retrieve pathological sequences pointers indeces.

def retrieve_pathological_seq(path):

    paths = sorted(glob.glob(path + "/*.jpg")) 
    
    len_ = 0
    start = 0
    is_moving = False
    
    pathological_positions = []
    pathological_pointers = []
    
    for img_path in paths:
    
      if (image_mode_check(img_path)=="RGB"):
        
        if is_moving == False:
          start = paths.index(img_path)
          len_ = 1
          is_moving = True
        else:
          len_ = len_ + 1
    
      else:
        if is_moving:
          pathological_positions.append((start, len_))
          pathological_pointers.append((start, (start + (len_ - 1)) + 1))
          len_ = 0
          start = 0
          is_moving = False
        else:
          continue
    print("[Done] retrieve pathological sequences pointers indeces")

    return paths, pathological_pointers, pathological_positions

def colorize_emergency_seq(paths, pathological_pointers, len_emergency_seq, model):

    # case 1: one pathsological sequence has been found
    
    if len(pathological_pointers)<=1:
    
      print("One pathological sequence has been detected...")
      print("------------ Pathological sequence ------------")
      #visualize_rgb(paths[pathological_pointers[0][0]:pathological_pointers[0][1]])
      max_ = maximum(len(paths[:pathological_pointers[0][0]]), len(paths[pathological_pointers[0][1]:-1]))
    
      for x in range(max_):
    
        if x <= len(paths[:pathological_pointers[0][0]]): # To colorize prior sequence
          im_i = paths[(pathological_pointers[0][0] - x)]
          im_i = colorized_rgb(im_i, model)
          #visualize_colorized_image(im_i)
    
        if x <= len(paths[pathological_pointers[0][1]:-1]): # To colorize next sequence
          im_j = paths[(pathological_pointers[0][1] + x)]
          im_j = colorized_rgb(im_j, model)
          #visualize_colorized_image(im_j)
    
    # case 2: more than one pathsological sequences to colorize for emergency use
    else:
    
      print(f"{len(pathological_pointers)} pathological sequences has been detected...")
      
      for sequence in pathological_pointers:
        
        #visualize_rgb(paths[(sequence[0]+1):sequence[1]]) 
        print(f"Colorizing the {sequence} is on progress..")
    
        for x in range(len_emergency_seq):
    
          if x <= len_emergency_seq:  
            
            im_i = paths[(sequence[0] - x)]
            im_i = colorized_rgb(im_i, model)
            #visualize_colorized_image(im_i) 
    
            im_j = paths[(sequence[1] + x)]
            im_j = colorized_rgb(im_j, model)
            #visualize_colorized_image(im_j)
    
    return print("[Done] Colorization of all emergency sequence.")

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