import omni.replicator.core as rep
import numpy as np

from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.syntheticdata import SyntheticData

cfg = {
      "output": 'C:/Users/student1/output',
      "colors": True,
      "frames": 2000,
      "subframes": 6,
      "format": "png"
}

OBJ = 'omniverse://localhost/assets/elem/'
ENV = 'omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/clean_cloudy_sky_and_floor.usd'

tables_dict = {
    'TABLE1' : 'omniverse://localhost/assets/tables/table_1.usd',
    'TABLE2' : 'omniverse://localhost/assets/tables/table_2.usd',
    'TABLE3' : 'omniverse://localhost/assets/tables/table_3.usd',
    'TABLE4' : 'omniverse://localhost/assets/tables/table_4.usd',
    'TABLE5' : 'omniverse://localhost/assets/tables/table_5.usd',
    'TABLE6' : 'omniverse://localhost/assets/tables/table_6.usd'
    }

classDict = {
    'BACKGROUND': 0,
    'obj': 1,
    'table': 2,
    'UNLABELLED': 3
}
            

# Generate the semantic filter predicate string from keys of classDict dict
predicate = 'class:'
for idx, classes in enumerate(classDict.keys()):
    if idx == 0:
        predicate = predicate + classes
    else:    
        predicate = predicate +  '|' + classes


# Set global semantic filter predicate
SyntheticData.Get().set_instance_mapping_semantic_filter(predicate)

# Modified from the replicator.core BasicWriter
class CustomWriter(Writer):
    def __init__(self, output_dir: str, classDict, colorize_semantic_segmentation: bool = True, image_format: str = "png"):
        self._frame_id = 0
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators = []
        self.colorize_semantic_segmentation = colorize_semantic_segmentation
        self.image_format = image_format
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        self.annotators.append(
            AnnotatorRegistry.get_annotator(
               "semantic_segmentation", init_params={"colorize": colorize_semantic_segmentation}
            )
        )

        self.CUSTOM_LABELS = classDict

    # modified from BasicWriter
    def write(self, data):
        self._write_img(data, "rgb")
        self._write_segmentation(data, "semantic_segmentation")
        self._frame_id += 1

    def _write_img(self, data, annotator: str):
        # Save the image under the correct path
        img_file_path = f"img_{self._frame_id}.{self.image_format}"
        self.backend.write_image(img_file_path, data[annotator])

    def _write_segmentation(self, data: dict, annotator: str):
        semantic_seg_data = data[annotator]["data"]
        height, width = semantic_seg_data.shape[:2]
        file_path = (
            f"semantic_segmentation_{self._frame_id}.{self.image_format}"
        )
        
        semantic_seg_data = semantic_seg_data.view(np.uint8).reshape(height, width, -1)
        self.backend.write_image(file_path, semantic_seg_data)
                    

    # Modified from omni.replicator.core.tools.colorize_segmentation
    # Same as that but instead of color mapping in shape (width, height, 4)
    # it maps custom int labels for seg mask in shape (width, height)
    def seg_data_as_labels(self, data, labels, mapping):
        unique_ids = np.unique(data)
        seg_as_labels = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
        for _id in enumerate(unique_ids):
            obj_label = [*labels[str(_id)].values()][0].lower()
            if obj_label in mapping:
                seg_as_labels[data == _id] = mapping[obj_label]

        return seg_as_labels


#End of the writer class
##########################################

WriterRegistry.register(CustomWriter)

with rep.new_layer():

    # randomize lights, sphere light instead of distant or dome in order to create shadows
    def sphere_lights():
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.choice([3000, 3500, 4000, 4500, 5000, 5500, 6000]),
            intensity=rep.distribution.uniform(20000, 70000),
            scale=200,
            position=(300, 700, -350),
            count=1
        )
        return lights.node


    # create a plane to sample objects on in randomize_obj
    # position(x,y,z) y is around the height of the tables,
    plane_samp = rep.create.plane(position=(0, 225, 0), scale=(2, 1, 1.5), visible=False)
    env = rep.create.from_usd(ENV)

    # create a camera
    camera = rep.create.camera(focus_distance=100, look_at=plane_samp, name="main_camera")

    # create a renderer, resolution is defined here
    rp = rep.create.render_product(camera, (512, 512))

    # Create the tables for variation
    tables = []
    for x in tables_dict:
        table_usd = rep.create.from_usd(tables_dict[x], semantics=[('class', 'table')])
        tables.append(table_usd)


    # function for scattering the instantiated prop prims on to the plane_samp
    def randomize_obj():
        obj = rep.randomizer.instantiate(
            rep.utils.get_usd_files(OBJ), 
            size=3, 
            with_replacements=False
            )

        with obj:
            rep.modify.pose(rotation=(0, 180, 0))
            rep.modify.semantics([("class", 'obj')])
            rep.randomizer.scatter_2d(plane_samp)
        return obj.node
    
    # Visibility distribution sequence for the table randomization
    one_sequence = [False] * len(tables)
    viz_matrix = []
    for x in range (len(tables)):
        arr = one_sequence.copy()
        arr[x] = True
        viz_matrix.append(arr)


    # Register defined randomization functions to the randomizer
    rep.randomizer.register(sphere_lights)
    rep.randomizer.register(randomize_obj)

    # Call the randomizer on each frame
    with rep.trigger.on_frame(num_frames=cfg["frames"], rt_subframes=cfg["subframes"]):
        for idx, table in enumerate(tables):
            with table:
                rep.modify.visibility(rep.distribution.sequence(viz_matrix[idx]))

        with camera:
            rep.modify.pose(position=rep.distribution.uniform((0, 300, -600), (1000, 1000, -200)), look_at=plane_samp)

        rep.randomizer.randomize_obj()
        rep.randomizer.sphere_lights()


# Initialize and attach writer
writer = rep.WriterRegistry.get("CustomWriter")
writer.initialize(output_dir=cfg["output"],
                  classDict=classDict,
                  colorize_semantic_segmentation=cfg["colors"],
                  image_format=cfg["format"])
writer.attach([rp])