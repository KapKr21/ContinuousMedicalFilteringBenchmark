import java.util.List;
import org.apache.log4j.BasicConfigurator;
import org.scify.jedai.blockbuilding.*;
import org.scify.jedai.blockprocessing.blockcleaning.ComparisonsBasedBlockPurging;
import org.scify.jedai.blockprocessing.comparisoncleaning.ComparisonPropagation;
import org.scify.jedai.datamodel.*;
import org.scify.jedai.datareader.entityreader.*;
import org.scify.jedai.datareader.groundtruthreader.*;
import org.scify.jedai.utilities.BlocksPerformance;
import org.scify.jedai.utilities.datastructures.*;

/**
 * Runs the Parameter-Free blocking workflow on all 8 medical datasets.
 * Uses Standard Blocking for most datasets, Q-Grams for UMLS (ontology data).
 *
 * Compile & run from repo root:
 *   javac -cp "blockingWorkflows/lib/*" -d out MedicalBFParameterFreeRunner.java
 *   java  -cp "out:blockingWorkflows/lib/*" MedicalBFParameterFreeRunner
 */

public class MedicalBFParameterFreeRunner {

    static final String DIR = "blockingWorkflows/data/medical/";
    static final int QGRAM_SIZE = 6;   // q=6 for UMLS/RxNorm: reduces block sizes dramatically

    // [label, D1 file, D2 file, GT file, use_qgrams]
    static final String[][] DATASETS = {
        // ── FEBRL: dirty ER (deduplication within one collection) ──
        {"FEBRL-1 (Dirty ER, 1k)",   "febrl1ProfilesD1",      "febrl1ProfilesD2",      "febrl1Duplicates",      "false"},
        {"FEBRL-2 (Dirty ER, 5k)",   "febrl2ProfilesD1",      "febrl2ProfilesD2",      "febrl2Duplicates",      "false"},
        {"FEBRL-3 (Dirty ER, 5k)",   "febrl3ProfilesD1",      "febrl3ProfilesD2",      "febrl3Duplicates",      "false"},
        // ── FEBRL-4: clean-clean ER ──
        {"FEBRL-4 (Clean-Clean, 5k)","febrl4ProfilesD1",      "febrl4ProfilesD2",      "febrl4Duplicates",      "false"},
        // ── Synthea: dirty ER (merged collection) ──
        {"Synthea (Dirty ER, 6k)",   "syntheaProfilesD1",     "syntheaProfilesD2",     "syntheaDuplicates",     "false"},
        // ── Clean-Clean datasets ──
        {"MedMentions (22k)",        "medmentionsProfilesD1", "medmentionsProfilesD2", "medmentionsDuplicates", "false"},
        {"CMS (64k)",                "cmsProfilesD1",         "cmsProfilesD2",         "cmsDuplicates",         "false"},
        // ── UMLS: uses Q-Grams because ontology terms are short/ambiguous ──
        {"UMLS (135k, Q-Grams)",     "umlsProfilesD1",        "umlsProfilesD2",        "umlsDuplicates",        "true"},
        {"RxNorm (Q-Grams)",         "rxnormProfilesD1",      "rxnormProfilesD2",      "rxnormDuplicates",      "true"},
    };

    public static void main(String[] args) {
        BasicConfigurator.configure();

        System.out.println("Medical Datasets — Parameter-Free Blocking Workflow\n");

        for (String[] ds : DATASETS) {
            String label    = ds[0];
            String pathD1   = DIR + ds[1];
            String pathD2   = DIR + ds[2];
            String pathGt   = DIR + ds[3];
            boolean qgrams  = Boolean.parseBoolean(ds[4]);

            System.out.println("Dataset : " + label);

            try {
                // Load profiles and ground truth
                List<EntityProfile> p1 = new EntitySerializationReader(pathD1).getEntityProfiles();
                List<EntityProfile> p2 = new EntitySerializationReader(pathD2).getEntityProfiles();
                AbstractDuplicatePropagation gt = new BilateralDuplicatePropagation(
                        new GtSerializationReader(pathGt).getDuplicatePairs(null));

                System.out.printf("D1: %,d  |  D2: %,d  |  GT pairs: %,d%n",
                        p1.size(), p2.size(), gt.getDuplicates().size());

                // Block Building
                List<AbstractBlock> blocks;
                long t1 = System.currentTimeMillis();

                if (qgrams) {
                    // Q-Grams blocking: better for short ontology/drug terms
                    QGramsBlocking qb = new QGramsBlocking(QGRAM_SIZE);
                    blocks = qb.getBlocks(p1, p2);
                    System.out.println("Block builder: Q-Grams (q=" + QGRAM_SIZE + ")");
                } else {
                    // Standard token blocking: better for natural-language fields
                    blocks = new StandardBlocking().getBlocks(p1, p2);
                    System.out.println("Block builder: Standard Blocking");
                }

                long t2 = System.currentTimeMillis();

                // Block Purging (parameter-free)
                blocks = new ComparisonsBasedBlockPurging(true).refineBlocks(blocks);
                long t3 = System.currentTimeMillis();

                // Comparison Propagation (parameter-free)
                blocks = new ComparisonPropagation().refineBlocks(blocks);
                long t4 = System.currentTimeMillis();

                // Statistics
                BlocksPerformance stats = new BlocksPerformance(blocks, gt);
                stats.setStatistics();
                stats.printStatistics(0, "", "");

                System.out.printf("Block Building time : %,d ms%n", t2 - t1);
                System.out.printf("Block Purging time  : %,d ms%n", t3 - t2);
                System.out.printf("Meta-Blocking time  : %,d ms%n", t4 - t3);
                System.out.printf("Total time          : %,d ms%n", t4 - t1);

            } catch (Exception e) {
                System.out.println("ERROR: " + e.getMessage());
                e.printStackTrace();
            }
            System.out.println();
        }

        System.out.println("All datasets done");
    }
}
