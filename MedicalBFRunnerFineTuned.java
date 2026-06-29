import java.util.List;
import org.apache.log4j.BasicConfigurator;
import org.scify.jedai.blockbuilding.*;
import org.scify.jedai.blockprocessing.IBlockProcessing;
import org.scify.jedai.blockprocessing.blockcleaning.BlockFiltering;
import org.scify.jedai.blockprocessing.blockcleaning.ComparisonsBasedBlockPurging;
import org.scify.jedai.blockprocessing.comparisoncleaning.ComparisonPropagation;
import org.scify.jedai.datamodel.*;
import org.scify.jedai.datareader.entityreader.*;
import org.scify.jedai.datareader.groundtruthreader.*;
import org.scify.jedai.utilities.BlocksPerformance;
import org.scify.jedai.utilities.datastructures.*;
import org.scify.jedai.utilities.enumerations.ComparisonCleaningMethod;

/**
 * Parameterized Blocking Workflow on Medical Datasets.
 *
 * For each dataset, tries multiple configurations:
 *   - Block Building: Standard Blocking OR Q-Grams with q = 3,4,5,6
 *   - Block Purging: ON / OFF
 *   - Block Filtering: with different ratios (numbered configs 0..39 map to ratios 0.025..0.99)
 *   - Meta-Blocking: 5 algorithms x multiple weighting configs
 *
 * Prints PC/PQ/F1 for each configuration so you can identify the best one per dataset.
 *
 * Compile:
 *   javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunnerFineTuned.java
 *
 * Run:
 *   java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned
 *
 * To run a single dataset (by index 0-8):
 *   java -Xmx12g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerFineTuned 3
 */
public class MedicalBFRunnerFineTuned {

    static final String DIR = "blockingWorkflows/data/medical/";

    // Dataset definitions: {label, D1, D2, GT, useQGrams("true"/"false")}
    static final String[][] DATASETS = {
        {"FEBRL-1 (Dirty ER)",       "febrl1ProfilesD1",      "febrl1ProfilesD2",      "febrl1Duplicates",      "false"},
        {"FEBRL-2 (Dirty ER)",       "febrl2ProfilesD1",      "febrl2ProfilesD2",      "febrl2Duplicates",      "false"},
        {"FEBRL-3 (Dirty ER)",       "febrl3ProfilesD1",      "febrl3ProfilesD2",      "febrl3Duplicates",      "false"},
        {"FEBRL-4 (Clean-Clean)",    "febrl4ProfilesD1",      "febrl4ProfilesD2",      "febrl4Duplicates",      "false"},
        {"Synthea (Dirty ER)",       "syntheaProfilesD1",     "syntheaProfilesD2",     "syntheaDuplicates",     "false"},
        {"MedMentions (Clean-Clean)","medmentionsProfilesD1", "medmentionsProfilesD2", "medmentionsDuplicates", "false"},
        {"CMS (Clean-Clean)",        "cmsProfilesD1",         "cmsProfilesD2",         "cmsDuplicates",         "false"},
        {"UMLS (Clean-Clean)",       "umlsProfilesD1",        "umlsProfilesD2",        "umlsDuplicates",        "true"},
        {"RxNorm (Clean-Clean)",     "rxnormProfilesD1",      "rxnormProfilesD2",      "rxnormDuplicates",      "true"},
    };

    // Meta-blocking algorithms to try
    static final ComparisonCleaningMethod[] MB_METHODS = {
        ComparisonCleaningMethod.WEIGHTED_EDGE_PRUNING,                  // WEP
        ComparisonCleaningMethod.CARDINALITY_EDGE_PRUNING,               // CEP
        ComparisonCleaningMethod.RECIPROCAL_CARDINALITY_NODE_PRUNING,    // RCNP
        ComparisonCleaningMethod.BLAST,                                  // BLAST
    };

    // Meta-blocking weighting configs to try (0-5 covers the main weighting schemes)
    static final int[] MB_CONFIGS = {0, 1, 2, 3, 4, 5};

    // Block Filtering configs to try (each maps to a ratio)
    // Config 0 = 0.025, 5 = 0.15, 10 = 0.275, 15 = 0.40, 20 = 0.525, 25 = 0.65, 30 = 0.775, 35 = 0.90
    static final int[] BF_CONFIGS = {5, 10, 15, 20, 25, 30, 35};

    // Q-gram sizes to try for Q-Gram datasets
    static final int[] QGRAM_SIZES = {3, 4, 5, 6};

    public static void main(String[] args) {
        BasicConfigurator.configure();

        int startDs = 0;
        int endDs   = DATASETS.length;

        // Allow running a single dataset by index
        if (args.length > 0) {
            startDs = Integer.parseInt(args[0]);
            endDs   = startDs + 1;
        }

        System.out.println("Medical Datasets — Parameterized Blocking Workflow Tuning");

        for (int dsIdx = startDs; dsIdx < endDs; dsIdx++) {
            String[] ds    = DATASETS[dsIdx];
            String label   = ds[0];
            String pathD1  = DIR + ds[1];
            String pathD2  = DIR + ds[2];
            String pathGt  = DIR + ds[3];
            boolean qgrams = Boolean.parseBoolean(ds[4]);

            System.out.println("DATASET [" + dsIdx + "]: " + label);

            try {
                // Load profiles and ground truth
                List<EntityProfile> p1 = new EntitySerializationReader(pathD1).getEntityProfiles();
                List<EntityProfile> p2 = new EntitySerializationReader(pathD2).getEntityProfiles();
                AbstractDuplicatePropagation gt = new BilateralDuplicatePropagation(
                        new GtSerializationReader(pathGt).getDuplicatePairs(null));

                System.out.printf("D1: %,d  |  D2: %,d  |  GT: %,d%n%n",
                        p1.size(), p2.size(), gt.getDuplicates().size());

                // Track best config
                double bestF1 = -1;
                String bestConfig = "";

                // PHASE 1: Parameter-free baseline (Comparison Propagation)
                System.out.println("── Phase 1: Parameter-Free Baseline (Block Purging + Comparison Propagation) ──");

                if (qgrams) {
                    for (int q : QGRAM_SIZES) {
                        double[] result = runParameterFree(p1, p2, gt, true, q);
                        String cfg = String.format("Q-Grams(q=%d) + Purging + CompPropagation", q);
                        printResult(cfg, result);
                        if (result[2] > bestF1) { bestF1 = result[2]; bestConfig = cfg; }
                    }
                } else {
                    // Standard blocking
                    double[] result = runParameterFree(p1, p2, gt, false, 0);
                    String cfg = "Standard + Purging + CompPropagation";
                    printResult(cfg, result);
                    if (result[2] > bestF1) { bestF1 = result[2]; bestConfig = cfg; }

                    // Also try Q-Grams q=4,5,6 (might help for some datasets)
                    for (int q : new int[]{4, 5, 6}) {
                        result = runParameterFree(p1, p2, gt, true, q);
                        cfg = String.format("Q-Grams(q=%d) + Purging + CompPropagation", q);
                        printResult(cfg, result);
                        if (result[2] > bestF1) { bestF1 = result[2]; bestConfig = cfg; }
                    }
                }

                // PHASE 2: Block Filtering + Meta-Blocking (parameterized)
                System.out.println("\nPhase 2: Tuned (Block Purging + Block Filtering + Meta-Blocking)");

                // Determine which block builders to try
                int[][] blockBuilders; // {useQGrams(0/1), qgramSize}
                if (qgrams) {
                    blockBuilders = new int[][]{{1, 3}, {1, 4}, {1, 5}, {1, 6}};
                } else {
                    blockBuilders = new int[][]{{0, 0}, {1, 4}, {1, 5}};
                }

                for (int[] bb : blockBuilders) {
                    boolean useQ = bb[0] == 1;
                    int qSize    = bb[1];
                    String bbName = useQ ? "Q-Grams(q=" + qSize + ")" : "Standard";

                    // Build blocks once for this block builder
                    List<AbstractBlock> rawBlocks;
                    if (useQ) {
                        rawBlocks = new QGramsBlocking(qSize).getBlocks(p1, p2);
                    } else {
                        rawBlocks = new StandardBlocking().getBlocks(p1, p2);
                    }

                    if (rawBlocks == null || rawBlocks.isEmpty()) {
                        System.out.println("  [SKIP] " + bbName + " produced no blocks");
                        continue;
                    }

                    // Try with and without block purging
                    for (boolean purge : new boolean[]{true, false}) {
                        List<AbstractBlock> purgedBlocks;
                        if (purge) {
                            purgedBlocks = new ComparisonsBasedBlockPurging(true).refineBlocks(rawBlocks);
                        } else {
                            purgedBlocks = rawBlocks;
                        }

                        if (purgedBlocks == null || purgedBlocks.isEmpty()) continue;

                        // Try different Block Filtering configs
                        for (int bfConf : BF_CONFIGS) {
                            BlockFiltering bf = new BlockFiltering();
                            bf.setNumberedGridConfiguration(bfConf);
                            List<AbstractBlock> filteredBlocks;
                            try {
                                filteredBlocks = bf.refineBlocks(purgedBlocks);
                            } catch (Exception e) {
                                continue;
                            }
                            if (filteredBlocks == null || filteredBlocks.isEmpty()) continue;

                            // Try different Meta-Blocking methods and configs
                            for (ComparisonCleaningMethod mbMethod : MB_METHODS) {
                                for (int mbConf : MB_CONFIGS) {
                                    try {
                                        IBlockProcessing mb = ComparisonCleaningMethod.getDefaultConfiguration(mbMethod);
                                        mb.setNumberedGridConfiguration(mbConf);
                                        List<AbstractBlock> finalBlocks = mb.refineBlocks(filteredBlocks);

                                        if (finalBlocks == null || finalBlocks.isEmpty()) continue;

                                        BlocksPerformance stats = new BlocksPerformance(finalBlocks, gt);
                                        stats.setStatistics();

                                        double pc = stats.getPc();
                                        double pq = stats.getPq();
                                        double f1 = (pc + pq == 0) ? 0 : 2 * pc * pq / (pc + pq);

                                        String cfg = String.format("%s | Purge=%s | BF=%d | %s(cfg=%d)",
                                                bbName, purge, bfConf, mbMethod.name(), mbConf);

                                        // Only print if F1 improves over baseline or is top-tier
                                        if (f1 > bestF1 * 0.9 || f1 > 0.5) {
                                            printResult(cfg, new double[]{pc, pq, f1});
                                        }

                                        if (f1 > bestF1) {
                                            bestF1 = f1;
                                            bestConfig = cfg;
                                        }
                                    } catch (Exception e) {
                                        // Skip configs that cause errors (e.g., OOM on large datasets)
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                // SUMMARY for this dataset
                System.out.println("\nBEST CONFIG for " + label + " ");
                System.out.printf("Config: %s%n", bestConfig);
                System.out.printf("F1    : %.4f%n", bestF1);

            } catch (Exception e) {
                System.out.println("ERROR on dataset " + label + ": " + e.getMessage());
                e.printStackTrace();
            }
        }

        System.out.println("\nTuning complete");
    }

    /**
     * Run the parameter-free workflow: Block Building → Block Purging → Comparison Propagation
     * Returns: [PC, PQ, F1]
     */
    static double[] runParameterFree(List<EntityProfile> p1, List<EntityProfile> p2,
                                     AbstractDuplicatePropagation gt,
                                     boolean useQGrams, int qSize) {
        try {
            List<AbstractBlock> blocks;
            if (useQGrams) {
                blocks = new QGramsBlocking(qSize).getBlocks(p1, p2);
            } else {
                blocks = new StandardBlocking().getBlocks(p1, p2);
            }

            blocks = new ComparisonsBasedBlockPurging(true).refineBlocks(blocks);
            blocks = new ComparisonPropagation().refineBlocks(blocks);

            BlocksPerformance stats = new BlocksPerformance(blocks, gt);
            stats.setStatistics();

            double pc = stats.getPc();
            double pq = stats.getPq();
            double f1 = (pc + pq == 0) ? 0 : 2 * pc * pq / (pc + pq);
            return new double[]{pc, pq, f1};
        } catch (Exception e) {
            System.out.println("  [ERROR] " + e.getMessage());
            return new double[]{0, 0, 0};
        }
    }

    static void printResult(String config, double[] result) {
        System.out.printf("  %-70s | PC=%.4f  PQ=%.4f  F1=%.4f%n",
                config, result[0], result[1], result[2]);
    }
}
