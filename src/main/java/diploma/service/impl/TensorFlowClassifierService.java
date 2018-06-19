package diploma.service.impl;

import diploma.service.ClassifierService;
import diploma.utils.PredictedStyle;
import diploma.utils.StyleList;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

@Service("classifierService")
public class TensorFlowClassifierService implements ClassifierService {
    private static final String MODEL_DIR = "model";
    private static final String STATIC_DIR = "static";
    private static final String OUTPUT_GRAPH = "output_graph.pb";
    private static final String OUTPUT_LABELS = "output_labels.txt";
/*
    public void classifyV2(String imageFile) {
        byte[] graphDef = readAllBytesOrExit(getClass().getResource(MODEL_DIR + OUTPUT_GRAPH));
        List<String> labels = readAllLinesOrExit(getClass().getResource(MODEL_DIR + OUTPUT_LABELS));
        byte[] imageBytes = readAllBytesOrExit(getClass().getResource(REPOSITORY_DIR + imageFile));

        try (Tensor image = Tensor.create(imageBytes);
             Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("Placeholder", image).fetch("final_result").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                float[] labelProbabilities = ((Tensor<Float>)result).copyTo(new float[1][nlabels])[0];
                int[] topThreeIndexes = getTopThreeIndexes(labelProbabilities);
                System.out.println(String.format("BEST MATCH: %s (%.2f%% likely)",
                        StyleList.getStyle(Integer.parseInt(labels.get(topThreeIndexes[0]))),
                        labelProbabilities[topThreeIndexes[0]] * 100f));
                System.out.println(String.format("Second MATCH: %s (%.2f%% likely)",
                        StyleList.getStyle(Integer.parseInt(labels.get(topThreeIndexes[1]))),
                        labelProbabilities[topThreeIndexes[1]] * 100f));
                System.out.println(String.format("Third MATCH: %s (%.2f%% likely)",
                        StyleList.getStyle(Integer.parseInt(labels.get(topThreeIndexes[2]))),
                        labelProbabilities[topThreeIndexes[2]] * 100f));
            }
        }
    }
*/
    @Override
    public PredictedStyle[] classify(String imageFile) throws IOException{
        PredictedStyle[] predictions = new PredictedStyle[3];

        byte[] graphDef = readAllBytesOrExit(MODEL_DIR + File.separator + OUTPUT_GRAPH);
        List<String> labels = readAllLinesOrExit(MODEL_DIR + File.separator + OUTPUT_LABELS);
        byte[] imageBytes = readAllBytesOrExit(STATIC_DIR + File.separator + imageFile);

        try (Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, image);
            int[] topThreeIndexes = getTopThreeIndexes(labelProbabilities);

            for (int i = 0; i < 3; i++) {
                predictions[i] = new PredictedStyle(
                        StyleList.getStyle(Integer.parseInt(labels.get(topThreeIndexes[i]))),
                        labelProbabilities[topThreeIndexes[i]] * 100f
                );
            }
        }
        return predictions;
    }

    private int[] getTopThreeIndexes(final float[] probabilities) {
        float[] copy = probabilities.clone();
        int best = maxIndex(copy);
        copy[best] = 0;
        int second = maxIndex(copy);
        copy[second] = 0;
        int third = maxIndex(copy);

        return new int[] {best, second, third};
    }

    private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
        try (Graph g = new Graph()) {
            GraphBuilder b = new GraphBuilder(g);

            final int H = 299;
            final int W = 299;
            final float mean = 117f;
            final float scale = 1f;

            final Output<String> input = b.constant("input", imageBytes);
            final Output<Float> output =
                    b.div(
                            b.sub(
                                    b.resizeBilinear(
                                            b.expandDims(
                                                    b.cast(b.decodeJpeg(input, 3), Float.class),
                                                    b.constant("make_batch", 0)),
                                            b.constant("size", new int[] {H, W})),
                                    b.constant("mean", mean)),
                            b.constant("scale", scale));
            try (Session s = new Session(g)) {
                return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
            }
        }
    }

    private int maxIndex(final float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private float[] executeInceptionGraph(final byte[] graphDef, final Tensor<Float> image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            //g.operations().forEachRemaining(System.out::println);
            try (Session s = new Session(g);
                 Tensor<Float> result =
                         s.runner()
                                 .feed("module_apply_default/hub_input/Mul", image)
                                 .fetch("final_result")
                                 .run().get(0).expect(Float.class)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor " +
                                            "where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                float[][] test = new float[1][nlabels];
                result.copyTo(test);
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    private List<String> readAllLinesOrExit(final String path) throws IOException {
        File file = new ClassPathResource(path).getFile();
        return Files.readAllLines(Paths.get(file.getAbsolutePath()));
    }

    private byte[] readAllBytesOrExit(final String path) throws IOException {
        File file = new ClassPathResource(path).getFile();
        return Files.readAllBytes(Paths.get(file.getAbsolutePath()));
    }

    static class GraphBuilder {
        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output<Float> div(Output<Float> x, Output<Float> y) {
            return binaryOp("Div", x, y);
        }

        <T> Output<T> sub(Output<T> x, Output<T> y) {
            return binaryOp("Sub", x, y);
        }

        <T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
            return binaryOp3("ResizeBilinear", images, size);
        }

        <T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
            return binaryOp3("ExpandDims", input, dim);
        }

        <T, U> Output<U> cast(Output<T> value, Class<U> type) {
            DataType dtype = DataType.fromClass(type);
            return g.opBuilder("Cast", "Cast")
                    .addInput(value)
                    .setAttr("DstT", dtype)
                    .build()
                    .<U>output(0);
        }

        Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .<UInt8>output(0);
        }

        <T> Output<T> constant(String name, Object value, Class<T> type) {
            try (Tensor<T> t = Tensor.<T>create(value, type)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", DataType.fromClass(type))
                        .setAttr("value", t)
                        .build()
                        .<T>output(0);
            }
        }
        Output<String> constant(String name, byte[] value) {
            return this.constant(name, value, String.class);
        }

        Output<Integer> constant(String name, int value) {
            return this.constant(name, value, Integer.class);
        }

        Output<Integer> constant(String name, int[] value) {
            return this.constant(name, value, Integer.class);
        }

        Output<Float> constant(String name, float value) {
            return this.constant(name, value, Float.class);
        }

        private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
        }

        private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
        }
        private Graph g;
    }
}
