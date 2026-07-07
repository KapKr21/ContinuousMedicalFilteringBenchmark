import java.util.List;
import org.apache.log4j.BasicConfigurator;
import org.scify.jedai.blockbuilding.*;
import org.scify.jedai.blockprocessing.IBlockProcessing;
import org.scify.jedai.blockprocessing.blockcleaning.BlockFiltering;
import org.scify.jedai.blockprocessing.blockcleaning.ComparisonsBasedBlockPurging;
import org.scify.jedai.datamodel.*;
import org.scify.jedai.datareader.entityreader.*;
import org.scify.jedai.datareader.groundtruthreader.*;
import org.scify.jedai.utilities.BlocksPerformance;
import org.scify.jedai.utilities.datastructures.*;
import org.scify.jedai.utilities.enumerations.ComparisonCleaningMethod;

/**
 * LITE version of MedicalBFRunnerFineTuned — reduced grid specifically for CMS and UMLS.
 *
 * Changes vs full version:
 *   - Block Filtering: 3 configs instead of 7 (10, 20, 30)
 *   - Meta-Blocking: 3 methods (WEP, CEP, RCNP) — BLAST dropped (too slow)
 *   - Weighting: 3 configs (0, 2, 4 = ARCS, ECBS, EJS)
 *   - Block builders: reduced to 2 per dataset
 *
 * ~24 combos per dataset instead of ~1008. Should finish in ~30-45 min per dataset.
 *
 * Compile:
 *   javac -cp "blockingWorkflows/lib/*" -d out MedicalBFRunnerLite.java
 *
 * Run both:
 *   java -Xmx6g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerLite
 *
 * Run only CMS (index 0) or UMLS (index 1):
 *   java -Xmx6g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerLite 0
 *   java -Xmx6g -cp "out:blockingWorkflows/lib/*" MedicalBFRunnerLite 1
 */
public class MedicalBFRunnerLite {

    static final String DIR = "blockingWorkflows/data/medical/";

    // Only CMS and UMLS
    static final String[][] DATASETS = {
        {"CMS (Clean-Clean)",        "cmsProfilesD1",         "cmsProfilesD2",         "cmsDuplicates",         "false"},
        {"UMLS (Clean-Clean)",       "umlsProfilesD1",        "umlsProfilesD2",        "umlsDuplicates",        "true"},
    };

    // Mid: WEP, CEP, and RCNP (drop BLAST)
    static final ComparisonCleaningMethod[] MB_METHODS = {
        ComparisonCleaningMethod.WEIGHTED_EDGE_PRUNING,               // WEP
        ComparisonCleaningMethod.CARDINALITY_EDGE_PRUNING,            // CEP
        ComparisonCleaningMethod.RECIPROCAL_CARDINALITY_NODE_PRUNING, // RCNP
    };

    // 3 weighting configs: ARCS(0), ECBS(2), EJS(4)
    static final int[] MB_CONFIGS = {0, 2, 4};

    // 3 Block Filtering configs: low/mid/high
    static final int[] BF_CONFIGS = {10, 20, 30};

    public static void main(String[] args) {
        BasicConfigurator.configure();

        int startDs = 0;
        int endDs   = DATASETS.length;

        if (args.length > 0) {
            startDs = Integer.parseInt(args[0]);
            endDs   = startDs + 1;
        }

        System.out.println("Medical Datasets — LITE Parameterized Blocking Workflow Tuning");
        System.out.println("(Mid grid: 3 MB methods, 3 weights, 3 BF configs)");
        System.out.println();

        for (int dsIdx = startDs; dsIdx < endDs; dsIdx++) {
            String[] ds    = DATASETS[dsIdx];
            String label   = ds[0];
            String pathD1  = DIR + ds[1];
            String pathD2  = DIR + ds[2];
            String pathGt  = DIR + ds[3];
            boolean qgrams = Boolean.parseBoolean(ds[4]);

            System.out.println("DATASET [" + dsIdx + "]: " + label);

            try {
                List<EntityProfile> p1 = new EntitySerializationReader(pathD1).getEntityProfiles();
                List<EntityProfile> p2 = new EntitySerializationReader(pathD2).getEntityProfiles();
                AbstractDuplicatePropagation gt = new BilateralDuplicatePropagation(
                        new GtSerializationReader(pathGt).getDuplicatePairs(null));

                System.out.printf("D1: %,d  |  D2: %,d  |  GT: %,d%n%n",
                        p1.size(), p2.size(), gt.getDuplicates().size());

                double bestF1 = -1;
                String bestConfig = "";

                // Parameterized grid search (parameter-free baselines in MedicalBFRunner.java)
                System.out.println("Tuned Grid Search:");

                // Block builders: minimal set — only the best per type
                int[][] blockBuilders;
                if (qgrams) {
                    blockBuilders = new int[][]{{1, 5}, {1, 6}};  // Q-Grams q=5 and q=6
                } else {
                    blockBuilders = new int[][]{{0, 0}, {1, 5}};  // Standard + Q-Grams q=5
                }

                for (int[] bb : blockBuilders) {
                    boolean useQ = bb[0] == 1;
                    int qSize    = bb[1];
                    String bbName = useQ ? "Q-Grams(q=" + qSize + ")" : "Standard";

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

                    System.out.println("\n  Block Builder: " + bbName +
                            " | Raw blocks: " + rawBlocks.size());

                    for (boolean purge : new boolean[]{true, false}) {
                        List<AbstractBlock> purgedBlocks;
                        if (purge) {
                            purgedBlocks = new ComparisonsBasedBlockPurging(true).refineBlocks(rawBlocks);
                        } else {
                            purgedBlocks = rawBlocks;
                        }

                        if (purgedBlocks == null || purgedBlocks.isEmpty()) continue;

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

                                        if (f1 > bestF1 * 0.9 || f1 > 0.5) {
                                            printResult(cfg, new double[]{pc, pq, f1});
                                        }

                                        if (f1 > bestF1) {
                                            bestF1 = f1;
                                            bestConfig = cfg;
                                        }
                                    } catch (Exception e) {
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                // SUMMARY
                System.out.printf("BEST CONFIG for %-43s %n", label);
                System.out.printf("Config: %-51s %n", bestConfig);
                System.out.printf("F1    : %-51.4f %n", bestF1);
                System.out.println();

            } catch (Exception e) {
                System.out.println("ERROR on dataset " + label + ": " + e.getMessage());
                e.printStackTrace();
            }
        }

        System.out.println("Lite tuning complete.");
    }

    static void printResult(String config, double[] result) {
        System.out.printf("  %-75s | PC=%.4f  PQ=%.4f  F1=%.4f%n",
                config, result[0], result[1], result[2]);
    }
}
