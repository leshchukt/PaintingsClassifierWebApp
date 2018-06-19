package diploma.controller;

import diploma.dto.UploadedImage;
import diploma.service.ClassifierService;
import diploma.utils.PredictedStyle;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class UploadImageController {
    private static final String STATIC_PATH = "static";
    private static final String REPOSITORY_PATH = "repository";

    @Autowired
    private ClassifierService classifierService;

    @GetMapping(value = "/")
    public String uploadImageHandler(Model model) {
        UploadedImage uploadedImage = new UploadedImage();
        model.addAttribute("uploadedImage", uploadedImage);
        return "index";
    }

    @PostMapping(value = "/")
    public String uploadImageHandler(Model model,
                                     @ModelAttribute("uploadedImage") UploadedImage uploadedImage)
            throws IOException{
        File uploadRootDir = new ClassPathResource(STATIC_PATH + File.separator + REPOSITORY_PATH).getFile();;

        MultipartFile fileData = uploadedImage.getImageData();
        String imageName = fileData.getOriginalFilename();

        if (imageName != null && imageName.length() > 0) {
            File uploadedFile = new File(uploadRootDir.getAbsolutePath() + File.separator + imageName);
            try (BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(uploadedFile))) {
                stream.write(fileData.getBytes());
            }
        }
        String relativeImagePath = REPOSITORY_PATH + File.separator + imageName;
        model.addAttribute("uploadedFile", relativeImagePath);

        getStylePredictions(model, relativeImagePath);
        return "index";
    }

    private void getStylePredictions(final Model model, final String relativeImagePath) throws IOException{
        PredictedStyle[] predictions = classifierService.classify(relativeImagePath);
        model.addAttribute("topStyle", predictions[0]);
        model.addAttribute("secondStyle", predictions[1]);
        model.addAttribute("thirdStyle", predictions[2]);
    }
}
