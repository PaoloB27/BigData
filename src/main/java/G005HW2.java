import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G005HW2 {

    private static double[][] distances; // Pre-computed distances

    public static void main(String[] args) throws IOException {
        // Checking number of command line parameters
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: <file_name> <num_centers> <num_outliers>");
        }
        // Read parameters
        String filename = args[0]; // Dataset filename
        int k = Integer.parseInt(args[1]); // Number of centers
        int z = Integer.parseInt(args[2]); // Number of allowed outliers
        // Reading points
        ArrayList<Vector> inputPoints = readVectorsSeq(filename);
        // Creating weights vector
        ArrayList<Long> weights = new ArrayList<>();
        for (int i = 0; i < inputPoints.size(); i++)
            weights.add(1L);
        // Output solution
        System.out.println("Input size n = " + inputPoints.size()); // n = |P|
        System.out.println("Number of centers k = " + k); // k
        System.out.println("Number of outliers z = " + z); // z
        // Precomputing distances
        distances = new double[inputPoints.size()][inputPoints.size()];
        PrecomputeDistances(inputPoints);
        // Running k-centerOUT + time elapsed computation
        long startTime = System.nanoTime();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0);
        long elapsedTime = System.nanoTime() - startTime;
        // Objective function
        double objective = ComputeObjective(inputPoints, solution, z);
        // Output solution
        System.out.println("Objective function = " + objective); // Objective
        System.out.println("Time of SeqWeightedOutliers = " + elapsedTime / 1000000); // Elapsed time
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Input reading methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Distance Pre-computation
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    /*
    The distances among all the points in P are computed and put in an upper triangular matrix.
    Since distance(x, y) = distance(y, x) for all x, y in P, the lower triangular matrix is initialized symmetrically.
    */
    public static void PrecomputeDistances(ArrayList<Vector> P) {
        for (int i = 0; i < P.size(); i++) {
            for (int j = i; j < P.size(); j++) {
                double distance = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
                distances[i][j] = distance;
                // Distance is commutative
                distances[j][i] = distance;
            }
        }
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // SeqWeightedOutliers
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z,
                                                        double alpha) {
        // Computation print of r_min
        double min_distance = Double.POSITIVE_INFINITY;
        for (int i = 0; i < k + z + 1; i++)
            for (int j = i + 1; j < k + z + 1; j++)
                if (distances[i][j] < min_distance)
                    min_distance = distances[i][j];
        double r = min_distance / 2;
        System.out.println("Initial guess = " + r); // initial guess r
        // Initialization of the number of guesses for r
        int guessCount = 1;
        // Evaluation of S
        while (true) {
            //Initialization of Z, S and W_z
            ArrayList<Vector> Z = new ArrayList<>(P); // Z <- P
            ArrayList<Vector> S = new ArrayList<>(); // S <- 0
            double W_z = 0;
            for (Long w : W)
                W_z += w;
            // Evaluation of S with the current r
            while (S.size() < k && W_z > 0) {
                double max = 0;
                int newCenterIndex = -1;
                // Seeking the first center
                for (int i = 0; i < P.size(); i++) {
                    double ballWeight = 0;
                    // Computation of the ball-weight
                    for (int j = 0; j < Z.size(); j++)
                        // Checking that Z.get(j) has not been removed from Z yet
                        if (Z.get(j) != null)
                            if (distances[i][j] <= (1 + 2 * alpha) * r)
                                ballWeight += W.get(j);
                    // Update of max if necessary
                    if (ballWeight > max) {
                        max = ballWeight;
                        newCenterIndex = i;
                    }
                }
                // Insert newCenter in S and remove points outside the ball from Z
                S.add(P.get(newCenterIndex));
                for (int i = 0; i < Z.size(); i++) {
                    if (Z.get(i) != null) {
                        if (distances[newCenterIndex][i] <= (3 + 4 * alpha) * r) {
                            // Removal of Z.get(i) by setting it to null
                            Z.set(i, null);
                            W_z -= W.get(i);
                        }
                    }
                }
            }
            // Check if S is the final solution
            if (W_z <= z) {
                System.out.println("Final guess = " + r); // final guess r
                System.out.println("Number of guesses = " + guessCount); // SeqWeightedOutliers guess
                return S;
            } else {
                r = 2 * r;
                guessCount++;
            }
        }
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // ComputeObjective
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    /*
    To compute the value of the objective function, distances from S for all
    points x in P are computed and inserted in an ArrayList<Double> distances.
    Then distances is ordered in ascending order and the z largest distances
    are removed because related to outliers.
    Finally, the last element among the remaining ones is returned as value of
    the objective function.
    */
    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z) {
        ArrayList<Double> objective = new ArrayList<>();
        // Compute distances d(x,S) with x in P, and S centers
        for (int i = 0; i < P.size(); i++) {
            // Compute the minimal distance of the current point in P from the centers in S
            Iterator<Vector> iterator_S = S.iterator();
            double min_distance = distances[i][P.indexOf(iterator_S.next())];
            while (iterator_S.hasNext()) {
                double current_distance = distances[i][P.indexOf(iterator_S.next())];
                if (current_distance < min_distance)
                    min_distance = current_distance;
            }
            objective.add(min_distance);
        }
        // Removal of z outliers (z largest distances)
        objective.sort(Comparator.naturalOrder());
        objective.removeAll(objective.subList(objective.size() - z, objective.size()));
        // Selection of the objective function
        return objective.get(objective.size() - 1);
    }
}
