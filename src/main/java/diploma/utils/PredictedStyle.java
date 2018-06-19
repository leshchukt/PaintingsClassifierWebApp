package diploma.utils;

public class PredictedStyle {
    private String style;
    private float probability;

    public PredictedStyle(final String style, final float probability) {
        this.style = style;
        this.probability = probability;
    }

    @Override
    public String toString() {
        return style +
                " (" +
                String.format("%.2f", probability) +
                "%)";
    }
}
