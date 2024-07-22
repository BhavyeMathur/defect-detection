from shampoo import ShampooThenConditioner

# source = "path/to/image"
source = "data/shampoo_defect/HT-GE505GC-T1-C-Snapshot-20240617-175033-516-200675409423_JPG_jpg.rf.2d818ba8c5bbe29416eb5046bf40791f.jpg"

model = ShampooThenConditioner("models/08_07_negative_shampoo.pt")
model.to("mps")

results = model(source, sigma=2, low=0, high=13, threshold=80, gradient_filter=None)
print(results.yolo)  # output from YOLOv8
print(results.classifications)  # defect/non-defect classification from vertical model
print(results.crops[0].shape)  # crops contains a crop of all cuts in the shampoo
results.plot()
