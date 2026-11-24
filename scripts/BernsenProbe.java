// Headless adapter only. Compile together with the unmodified upstream Fiji plugin.
import fiji.threshold.Auto_Local_Threshold;
import ij.ImagePlus;
import ij.Prefs;
import ij.process.ByteProcessor;
import java.nio.file.*;
import java.util.Arrays;

public class BernsenProbe {
    public static void main(String[] args) throws Exception {
        int w = Integer.parseInt(args[2]), h = Integer.parseInt(args[3]), contrast = Integer.parseInt(args[4]);
        byte[] input = Files.readAllBytes(Paths.get(args[0]));
        int step = Math.multiplyExact(w,h);
        if (input.length == 0 || input.length % step != 0) throw new IllegalArgumentException("bad stack");
        byte[] output = new byte[input.length];
        Prefs.setThreads(1);
        for (int offset = 0; offset < input.length; offset += step) {
            byte[] plane = Arrays.copyOfRange(input, offset, offset+step);
            ImagePlus img = new ImagePlus("input", new ByteProcessor(w,h,plane,null));
            new Auto_Local_Threshold().exec(img,"Bernsen",5,contrast,0,true);
            System.arraycopy((byte[]) img.getProcessor().getPixels(),0,output,offset,step);
        }
        Files.write(Paths.get(args[1]),output,StandardOpenOption.CREATE_NEW);
        System.exit(0);
    }
}
