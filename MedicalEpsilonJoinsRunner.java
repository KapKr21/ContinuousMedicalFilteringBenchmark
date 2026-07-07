import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import java.util.*;
import org.scify.jedai.datamodel.EntityProfile;
import org.scify.jedai.datamodel.IdDuplicates;
import org.scify.jedai.datareader.entityreader.EntitySerializationReader;
import org.scify.jedai.datareader.groundtruthreader.GtSerializationReader;
import utilities.RepresentationModel;
import utilities.SimilarityFunction;
import utilities.Tokenizer;

/**
 * Runs the ε-join on all medical datasets.
 *
 * Compile from repo root:
 *   javac -cp "joins/lib/*:blockingWorkflows/lib/*" \
 *         -d out \
 *         joins/src/utilities/*.java \
 *         MedicalEpsilonJoinsRunner.java
 *
 * Run:
 *   java -Xmx12g -cp "out:joins/lib/*:blockingWorkflows/lib/*" MedicalEpsilonJoinsRunner
 */
public class MedicalEpsilonJoinsRunner {

    static final String DIR = "blockingWorkflows/data/medical/";

    static final Object[][] DATASETS = {
        // {label, D1, D2, GT, threshold, SimilarityFunction, Tokenizer}
        {"FEBRL-1",     "febrl1ProfilesD1",      "febrl1ProfilesD2",      "febrl1Duplicates",      0.5f, SimilarityFunction.COSINE_SIM,  Tokenizer.WHITESPACE},
        {"FEBRL-2",     "febrl2ProfilesD1",       "febrl2ProfilesD2",      "febrl2Duplicates",      0.5f, SimilarityFunction.COSINE_SIM,  Tokenizer.WHITESPACE},
        {"FEBRL-3",     "febrl3ProfilesD1",       "febrl3ProfilesD2",      "febrl3Duplicates",      0.5f, SimilarityFunction.COSINE_SIM,  Tokenizer.WHITESPACE},
        {"FEBRL-4",     "febrl4ProfilesD1",      "febrl4ProfilesD2",      "febrl4Duplicates",      0.5f, SimilarityFunction.COSINE_SIM,  Tokenizer.WHITESPACE},
        {"Synthea",     "syntheaProfilesD1",     "syntheaProfilesD2",     "syntheaDuplicates",     0.4f, SimilarityFunction.COSINE_SIM,  Tokenizer.WHITESPACE},
        {"MedMentions", "medmentionsProfilesD1", "medmentionsProfilesD2", "medmentionsDuplicates", 0.6f, SimilarityFunction.COSINE_SIM,  Tokenizer.CHARACTER_TRIGRAMS},
        {"CMS",         "cmsProfilesD1",         "cmsProfilesD2",         "cmsDuplicates",         0.7f, SimilarityFunction.JACCARD_SIM, Tokenizer.WHITESPACE},
        {"UMLS",        "umlsProfilesD1",        "umlsProfilesD2",        "umlsDuplicates",        0.6f, SimilarityFunction.COSINE_SIM,  Tokenizer.CHARACTER_TRIGRAMS},
        {"RxNorm",      "rxnormProfilesD1",      "rxnormProfilesD2",      "rxnormDuplicates",      0.5f, SimilarityFunction.COSINE_SIM,  Tokenizer.CHARACTER_TRIGRAMS},
    };

    public static void main(String[] args) throws Exception {
        System.out.println("Medical Datasets — ε-Join (Schema-Agnostic)\n");

        for (Object[] ds : DATASETS) {
            String label             = (String) ds[0];
            String pathD1            = DIR + ds[1];
            String pathD2            = DIR + ds[2];
            String pathGt            = DIR + ds[3];
            float  thresh            = (float) ds[4];
            SimilarityFunction simFn = (SimilarityFunction) ds[5];
            Tokenizer tokenizer      = (Tokenizer) ds[6];

            System.out.println("Dataset   : " + label);
            System.out.printf ("Threshold : %.2f  |  Sim: %s  |  Tokenizer: %s%n",
                    thresh, simFn, tokenizer);

            List<EntityProfile> source = new EntitySerializationReader(pathD1).getEntityProfiles();
            List<EntityProfile> target = new EntitySerializationReader(pathD2).getEntityProfiles();
            Set<IdDuplicates> gtDups   = new GtSerializationReader(pathGt).getDuplicatePairs(source, target);

            System.out.printf("D1: %,d  |  D2: %,d  |  GT: %,d%n",
                    source.size(), target.size(), gtDups.size());

            // Build inverted index on source (D1)
            int noOfSource  = source.size();
            int[] sourceFreq = new int[noOfSource];
            Map<String, TIntList> index = new HashMap<>();

            for (int i = 0; i < noOfSource; i++) {
                String text = RepresentationModel.getAttributeValue(source.get(i));
                Set<String> tokens = RepresentationModel.tokenizeEntity(text, tokenizer);
                sourceFreq[i] = tokens.size();
                for (String token : tokens) {
                    index.computeIfAbsent(token, k -> new TIntArrayList()).add(i);
                }
            }

            // Query with target (D2)
            long t1      = System.currentTimeMillis();
            int[] counters = new int[noOfSource];
            int[] flags    = new int[noOfSource];
            Arrays.fill(flags, -1);
            List<int[]> candidates = new ArrayList<>();

            for (int targetId = 0; targetId < target.size(); targetId++) {
                String text = RepresentationModel.getAttributeValue(target.get(targetId));
                Set<String> tokens = RepresentationModel.tokenizeEntity(text, tokenizer);
                if (tokens.isEmpty()) continue;

                TIntSet cands = new TIntHashSet();
                for (String token : tokens) {
                    TIntList srcs = index.get(token);
                    if (srcs == null) continue;
                    for (TIntIterator it = srcs.iterator(); it.hasNext(); ) {
                        int srcId = it.next();
                        cands.add(srcId);
                        if (flags[srcId] != targetId) {
                            counters[srcId] = 0;
                            flags[srcId]    = targetId;
                        }
                        counters[srcId]++;
                    }
                }

                for (TIntIterator it = cands.iterator(); it.hasNext(); ) {
                    int srcId  = it.next();
                    float common = counters[srcId];
                    float sim;
                    switch (simFn) {
                        case COSINE_SIM:
                            sim = common / (float) Math.sqrt((float) sourceFreq[srcId] * tokens.size());
                            break;
                        case JACCARD_SIM:
                            sim = common / (sourceFreq[srcId] + tokens.size() - common);
                            break;
                        default:
                            sim = 2 * common / (sourceFreq[srcId] + tokens.size());
                    }
                    if (sim >= thresh) {
                        candidates.add(new int[]{srcId, targetId});
                    }
                }
            }
            long t2 = System.currentTimeMillis();

            // Evaluate
            long detected = 0;
            for (int[] pair : candidates) {
                if (gtDups.contains(new IdDuplicates(pair[0], pair[1]))) detected++;
            }

            long   totalC    = candidates.size();
            double pc        = gtDups.isEmpty() ? 0 : (double) detected / gtDups.size();
            double pq        = totalC == 0 ? 0 : (double) detected / totalC;
            double f1        = (pc + pq == 0) ? 0 : 2 * pc * pq / (pc + pq);
            long   bruteForce = (long) source.size() * target.size();
            double rr        = 1.0 - (double) totalC / bruteForce;

            System.out.printf("Candidates   : %,d%n", totalC);
            System.out.printf("Detected     : %,d%n", detected);
            System.out.printf("PC (Recall)  : %.4f%n", pc);
            System.out.printf("PQ (Precis.) : %.4f%n", pq);
            System.out.printf("F-Measure    : %.4f%n", f1);
            System.out.printf("Reduction    : %.4f%n", rr);
            System.out.printf("Query time   : %,d ms%n%n", t2 - t1);
        }
        System.out.println("All datasets done");
    }
}
