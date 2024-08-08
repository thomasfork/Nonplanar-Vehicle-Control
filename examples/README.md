This folder provides examples of how the ```vehicle_3d``` package may be used. An outline of key examples is given here:

### ```models/```

This folder provides several examples of vehicle models that are available through the python package. In particualar, scripts ending in ```_dof.py``` are interactive visual applications that allow the user to manually adjust the kinematic degrees of freedom of each model, such as adjusting the camber angle of a motorcycle and translating it on a surface.

### Examples pertaining to papers

Main scripts for several papers are available in the ```paper_examples``` folder. These indicate the paper to which they apply. 

For the publication available at https://arxiv.org/abs/2204.10446 please see the earlier commit https://github.com/thomasfork/Nonplanar-Vehicle-Control/tree/2a7992c540ec365f3840d32679159e2a0f37df2e

### ```planning/```

Several examples of planning problems set up with nonplanar vehicle models.

Point to point planning emphasizes navigating through off-road terrain, with or without obstacles, and an example of each is provided.

Speed planning demonstrates a planner similar to the safety system in https://arxiv.org/abs/2406.01724, where a known spatial reference curve is taken and safe operating speed is determined along the reference. The examples here maximize vehicle speed.


### ```racelines/```

Examples of racelines computed on several different surfaces with different models.
