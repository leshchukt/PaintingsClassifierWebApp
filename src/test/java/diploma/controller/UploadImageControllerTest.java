package diploma.controller;

import java.io.File;
import java.io.IOException;

import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import static org.junit.Assert.*;

public class UploadImageControllerTest {
    private static final String STATIC_PATH = "static";
    private static final String REPOSITORY_PATH = "repository";
    private static final String MODEL_PATH = "model";

    @Test
    public void testFilePath() throws IOException {
        ClassPathResource classPathResource = new ClassPathResource(MODEL_PATH);
        System.out.println(classPathResource.getFilename());
        File file = classPathResource.getFile();
        System.out.println(file.getAbsolutePath());
    }
}