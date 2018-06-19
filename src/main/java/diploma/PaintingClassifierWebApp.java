package diploma;

import diploma.service.impl.TensorFlowClassifierService;

import java.io.File;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class PaintingClassifierWebApp {

    public static void main(String[] args) {
        SpringApplication.run(PaintingClassifierWebApp.class, args);
    }
}
