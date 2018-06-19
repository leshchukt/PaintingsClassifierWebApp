package diploma.dto;

import org.springframework.web.multipart.MultipartFile;

public class UploadedImage {
    private MultipartFile imageData;

    public MultipartFile getImageData() {
        return imageData;
    }

    public void setImageData(final MultipartFile imageData) {
        this.imageData = imageData;
    }
}
