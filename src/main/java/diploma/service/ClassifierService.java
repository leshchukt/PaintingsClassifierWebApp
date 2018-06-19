package diploma.service;

import diploma.utils.PredictedStyle;

import java.io.IOException;

public interface ClassifierService {
    PredictedStyle[] classify(String imageFile) throws IOException;
}
