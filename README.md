## Planning of the DL reproducibility project

##### Group agreement:
2 joint appointments per week (1 physical and 1 hybrid). 
Deliverable code in week 8 (30th march)
Training/optimization in week 8/9
Blog writing in week 8/9/10
**Hand-in deadline 14 april**


##### Workload planning/distribution:
**Dataloader:**
Thomas
Lucas

**Model reconstruction:**
Thomas

**Extracting 3D patches:**


**Loss functions:**
Lucas 
Emma
Storm

**Inferencing:**


**Google Colab setup:**

**Blog Writing**:
Emma


### Logbook
**- Week 3:** 
Individual: 
Reading paper

Joint:
Discussion of topic and deep learning methods; 
Slide deck for intro meeting with Prerak Modi;
Kickoff meeting with Prerak Modi

Issues identified: 

**- Week 4:** 
Individual:
Acquaintance with 3DSlicer software, exploration of data format/3D images
Thomas/Lucas: setup initial python skeleton/start dataloader
Lucas/Storm: initial look into loss functions

Joint: discuss unclear parameters, debunk U-Net network structure further.

Issues identified: 
1. Dimensionality of convolutions is unknown which seems to make a mismatch
	- Posted questions to Prerak's doc how to approach this.
2. Dataset masks seem to be combined in one channel in Prerak's version of the data.
	- Discussed whether to use the original data set.  

**- Week 5**
Individual:

Joint:

Issues identified:
1. Current focal and dice loss do not average over all masks in current implementation (as a result of masks being in a single file). 
	- Fixing the Single Mask Channel problem could fix this. 

**- Week 6**
Individual:

Joint:

Issues identified:

