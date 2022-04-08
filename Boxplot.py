# libraries
import matplotlib.pyplot as plt

organs = [
    'Background',
    'Brain Stem',
    'Opt. Chiasm',
    'Mandible',
    'Opt. Ner. L',
    'Opt. Ner. R',
    'Parotid L',
    'Parotid R',
    'Subman. L',
    'Subman. R'
]

# Creating dataset
data_1 = [99.87812805175781, 99.87369537353516, 99.94606018066406, 99.93517303466797, 99.91661071777344,
          99.9234619140625, 99.90424346923828, 99.90361785888672, 99.94754791259766, 99.95294189453125]
data_2 = [79.27519989013672, 65.62236785888672, 51.36476135253906, 78.96709442138672, 73.89044189453125,
          25.321102142333984, 57.78175735473633, 12.358882904052734, 76.0916976928711, 85.28553771972656]
data_3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
data_4 = [65.21202850341797, 79.41487121582031, 81.61241149902344, 79.80852508544922, 71.10395812988281,
          82.31537628173828, 80.35713958740234, 72.05692291259766, 82.16178131103516, 80.72369384765625]
data_5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
data_6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
data_7 = [80.6937484741211, 73.83023071289062, 82.15179443359375, 75.25530242919922, 68.40914916992188,
          71.61516571044922, 57.56724548339844, 54.87646484375, 68.20973205566406, 81.8565444946289]
data_8 = [71.92049407958984, 69.87572479248047, 69.02788543701172, 81.6805648803711, 78.38492584228516,
          74.59222412109375, 59.19125747680664, 67.46054077148438, 70.84148406982422, 83.11048889160156]
data_9 = [71.38331604003906, 76.71102905273438, 73.29376983642578, 45.6716423034668, 40.28605651855469,
          46.82539749145508, 2.7397260665893555, 0.0, 61.51723861694336, 54.062496185302734]
data_10 = [64.2297592163086, 60.89266586303711, 56.06694793701172, 69.0240478515625, 40.25316619873047,
           20.833332061767578, 21.153846740722656, 51.82291793823242, 63.05084991455078, 45.942230224609375]

data = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10]

# figure related code
fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(data)

ax.set_title('Boxplot DSC per Organ', fontsize=14, fontweight='bold')
ax.set_xticklabels(organs, rotation=25, fontsize=10)
ax.set_xlabel('Organs')
ax.set_ylabel('DSC')

plt.show()
