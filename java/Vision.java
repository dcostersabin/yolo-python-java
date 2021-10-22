import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class Vision {

    private final String model_weights;
    private final String model_config;
    private final String current_dir;
    private final String class_file_name_dir;
    private final String output_path;
    private final List<String> classes;
    private final List<String> output_layers;
    private String input_path;
    private List<String> layer_names;
    private Net network;
    private Size size;
    private Integer height;
    private Integer width;
    private Integer channels;
    private Scalar mean;
    private Mat image;
    private Mat blob;
    private List<Mat> outputs;
    private List<Rect2d> boxes;
    private List<Float> confidences;
    private List<Integer> class_ids;
    private String outputFileName;
    private boolean save;
    private boolean errors;

    public Vision(String inputPath, String outputPath, Integer image_size, String outputFileName) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        this.input_path = inputPath;
        this.output_path = outputPath;
        this.outputFileName = outputFileName;
        boxes = new ArrayList<>();
        classes = new ArrayList<>();
        class_ids = new ArrayList<>();
        layer_names = new ArrayList<>();
        confidences = new ArrayList<>();
        double[] means = {0.0, 0.0, 0.0};
        mean = new Scalar(means);
        output_layers = new ArrayList<>();
        size = new Size(image_size, image_size);
        current_dir = System.getProperty("user.dir");
        model_weights = current_dir + "/Assets/models/yolov3-608.weights";
        model_config = current_dir + "/Assets/models/yolov3-608.cfg";
        class_file_name_dir = current_dir + "/Assets/models/coco.names";
        save = true;

    }

    private static int argmax(List<Float> array) {
        float max = array.get(0);
        int re = 0;
        for (int i = 1; i < array.size(); i++) {
            if (array.get(i) > max) {
                max = array.get(i);
                re = i;
            }
        }
        return re;
    }

    private void setClasses() {
        try {
            File f = new File(class_file_name_dir);
            Scanner reader = new Scanner(f);
            while (reader.hasNextLine()) {
                String class_name = reader.nextLine();
                classes.add(class_name);
            }
        } catch (FileNotFoundException e) {
            errors = true;
        }
    }

    private void setNetwork() {
        network = Dnn.readNet(model_weights, model_config);
    }

    private void setUnconnectedLayers() {

        for (Integer i : network.getUnconnectedOutLayers().toList()) {
            output_layers.add(layer_names.get(i - 1));
        }
    }

    private void setLayerNames() {
        layer_names = network.getLayerNames();
    }

    private void loadImage() {
        Mat img = Imgcodecs.imread(input_path);
        Mat resizedImage = new Mat();
        Imgproc.resize(img, resizedImage, size, 0.9, 0.9);
        height = resizedImage.height();
        width = resizedImage.width();
        channels = resizedImage.channels();
        image = resizedImage;
    }

    private void detectObject() {
        Mat blob_from_image = Dnn.blobFromImage(image, 0.00392, size, mean, true, false);
        network.setInput(blob_from_image);
        outputs = new ArrayList<Mat>();
        network.forward(outputs, output_layers);
        blob = blob_from_image;
    }

    private void getBoxDimensions() {
        for (Mat output : outputs) {

            for (int i = 0; i < output.height(); i++) {
                Mat row = output.row(i);
                MatOfFloat temp = new MatOfFloat(row);
                List<Float> detect = temp.toList();
                List<Float> score = detect.subList(5, 85);
                int class_id = argmax(score);
                float conf = score.get(class_id);
                if (conf >= 0.4) {
                    int center_x = (int) (detect.get(0) * width);
                    int center_y = (int) (detect.get(1) * height);
                    int w = (int) (detect.get(2) * width);
                    int h = (int) (detect.get(3) * height);
                    int x = (center_x - w / 2);
                    int y = (center_y - h / 2);
                    Rect2d box = new Rect2d(x, y, w, h);
                    boxes.add(box);
                    confidences.add(conf);
                    class_ids.add(class_id);

                }

            }

        }
    }

    private void drawLabels() {
        double[] rgb = new double[]{255, 255, 0};
        Scalar color = new Scalar(rgb);
        MatOfRect2d mat = new MatOfRect2d();
        mat.fromList(boxes);
        MatOfFloat confidence = new MatOfFloat();
        confidence.fromList(confidences);
        MatOfInt indices = new MatOfInt();
        int font = Imgproc.FONT_HERSHEY_PLAIN;
        Dnn.NMSBoxes(mat, confidence, (float) (0.4), (float) (0.4), indices);
        List indices_list = indices.toList();
        for (int i = 0; i < boxes.size(); i++) {
            if (indices_list.contains(i)) {
                if (save) {
                    Rect2d box = boxes.get(i);
                    Point x_y = new Point(box.x, box.y);
                    Point w_h = new Point(box.x + box.width, box.y + box.height);
                    Point text_point = new Point(box.x, box.y - 5);
                    Imgproc.rectangle(image, w_h, x_y, color);
                    String label = classes.get(class_ids.get(i));
                    Imgproc.putText(image, label, text_point, font, 1, color);
                }

            }
        }
        if (save) {
            Imgcodecs.imwrite(output_path + "\\" + outputFileName + ".png", image);
        }

    }


    public void loadPipeline() {
        try {
            setNetwork();
            setClasses();
            setLayerNames();
            setUnconnectedLayers();
            loadImage();
            detectObject();
            getBoxDimensions();
            drawLabels();
        } catch (Exception e) {
            errors = true;
        }

    }

}
