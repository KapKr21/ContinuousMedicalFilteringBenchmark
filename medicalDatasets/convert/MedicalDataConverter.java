package medicalDatasets.convert;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.apache.log4j.BasicConfigurator;
import org.scify.jedai.datamodel.EntityProfile;
import org.scify.jedai.datamodel.IdDuplicates;
import org.scify.jedai.datareader.entityreader.EntitySerializationReader;
import org.scify.jedai.datareader.entityreader.IEntityReader;

/**
 * MedicalDataConverter
 *
 * Converts the prepared medical CSV datasets into JedAI-compatible serialized
 * EntityProfile objects and ground-truth IdDuplicates sets.
 *
 * Input:  blockingWorkflows/data/medical/rawdata/  (CSV files)
 * Output: blockingWorkflows/data/medical/          (serialized .bin files)
 *
 * The output files are named to match the naming convention expected by the
 * blocking workflow Java experiments:
 *   <dataset>ProfilesD1   → Collection A entity profiles
 *   <dataset>ProfilesD2   → Collection B entity profiles
 *   <dataset>Duplicates   → Ground truth duplicate pairs
 *
 * Supported datasets (6):
 *   febrl1, febrl2, febrl3          (dirty ER – single collection dedup)
 *   febrl4                          (clean-clean ER – A/B split)
 *   synthea                         (dirty ER)
 *   medmentions                     (clean-clean ER)
 *   cms                             (clean-clean ER)
 *   umls                            (clean-clean ER)
 *   rxnorm                          (clean-clean ER)
 *
 * Compile from repo root:
 *   javac -cp "blockingWorkflows/lib/*" \
 *         -d out \
 *         medicalDatasets/convert/MedicalDataConverter.java
 *
 * Run from repo root:
 *   java -cp "out:blockingWorkflows/lib/*" \
 *        medicalDatasets.convert.MedicalDataConverter
 */

public class MedicalDataConverter {

    // Paths
    private static final String RAW_DIR = "blockingWorkflows/data/medical/rawdata/";
    private static final String OUT_DIR = "blockingWorkflows/data/medical/";

    // Dataset descriptors
    /**
     * Each descriptor encodes one conversion task.
     * Fields: name, fileA, fileB (null for dirty ER), gtFile, idColA, idColB
     */
    private static class Dataset {
        final String name;
        final String fileA;
        final String fileB;   // null = dirty ER (same file used as both collections)
        final String gtFile;

        Dataset(String name, String fileA, String fileB, String gtFile) {
            this.name   = name;
            this.fileA  = fileA;
            this.fileB  = fileB;
            this.gtFile = gtFile;
        }
    }

    private static final Dataset[] DATASETS = {
        // FEBRL dirty ER (single-collection deduplication)
        new Dataset("febrl1",      "febrl1.csv",         null,              "febrl1_groundtruth.csv"),
        new Dataset("febrl2",      "febrl2.csv",         null,              "febrl2_groundtruth.csv"),
        new Dataset("febrl3",      "febrl3.csv",         null,              "febrl3_groundtruth.csv"),
        // FEBRL 4 clean-clean
        new Dataset("febrl4",      "febrlA.csv",         "febrlB.csv",      "febrl4_groundtruth.csv"),
        // Synthea dirty ER — use the merged collection (originals + noisy copies)
        // as BOTH D1 and D2 so within-collection duplicates can be found
        new Dataset("synthea",     "syntheaB_with_dups.csv", null, "synthea_groundtruth.csv"),
        // MedMentions clean-clean
        new Dataset("medmentions", "medmentionsA.csv",   "medmentionsB.csv","medmentions_groundtruth.csv"),
        // CMS clean-clean
        new Dataset("cms",         "cmsA.csv",           "cmsB.csv",        "cms_groundtruth.csv"),
        // UMLS clean-clean
        new Dataset("umls",        "umlsA.csv",          "umlsB.csv",       "umls_groundtruth.csv"),
        // RxNorm clean-clean
        new Dataset("rxnorm",      "rxnormA.csv",        "rxnormB.csv",     "rxnorm_groundtruth.csv"),
    };

    // Main
    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        File outDir = new File(OUT_DIR);
        if (!outDir.exists()) outDir.mkdirs();

        System.out.println("=== MedicalDataConverter ===");
        System.out.println("Input  : " + new File(RAW_DIR).getAbsolutePath());
        System.out.println("Output : " + outDir.getAbsolutePath());
        System.out.println();

        int converted = 0;
        int skipped   = 0;

        for (Dataset ds : DATASETS) {
            System.out.println("── " + ds.name.toUpperCase() + " ──");

            File fA  = new File(RAW_DIR + ds.fileA);
            File fGt = new File(RAW_DIR + ds.gtFile);

            if (!fA.exists()) {
                System.out.println("  SKIP – missing: " + fA.getName());
                skipped++;
                continue;
            }
            if (!fGt.exists()) {
                System.out.println("  SKIP – missing: " + fGt.getName());
                skipped++;
                continue;
            }

            boolean isCleanClean = (ds.fileB != null);
            if (isCleanClean) {
                File fB = new File(RAW_DIR + ds.fileB);
                if (!fB.exists()) {
                    System.out.println("  SKIP – missing: " + fB.getName());
                    skipped++;
                    continue;
                }
                convertCleanClean(ds);
            } else {
                convertDirtyER(ds);
            }
            converted++;
        }

        System.out.println();
        System.out.println("=== Done: " + converted + " converted, " + skipped + " skipped ===");

        // Verify by re-reading one output file
        System.out.println("\nVerification (febrl1):");
        verifySerialized("febrl1");
    }

    // Clean-Clean ER conversion
    private static void convertCleanClean(Dataset ds) throws Exception {
        System.out.println("  Mode: clean-clean ER");

        List<EntityProfile> profilesA = readProfiles(RAW_DIR + ds.fileA);
        List<EntityProfile> profilesB = readProfiles(RAW_DIR + ds.fileB);

        System.out.printf("  Profiles A: %,d%n", profilesA.size());
        System.out.printf("  Profiles B: %,d%n", profilesB.size());

        // Build id → list-index maps for ground truth resolution
        Map<String, Integer> idxA = buildIndexMap(profilesA);
        Map<String, Integer> idxB = buildIndexMap(profilesB);

        Set<IdDuplicates> gt = readGroundTruth(RAW_DIR + ds.gtFile, idxA, idxB);
        System.out.printf("  GT pairs:   %,d%n", gt.size());

        // Serialize
        String outA  = OUT_DIR + ds.name + "ProfilesD1";
        String outB  = OUT_DIR + ds.name + "ProfilesD2";
        String outGt = OUT_DIR + ds.name + "Duplicates";

        writeProfiles(profilesA, outA);
        writeProfiles(profilesB, outB);
        writeGroundTruth(gt, outGt);

        System.out.println("  → " + new File(outA).getName());
        System.out.println("  → " + new File(outB).getName());
        System.out.println("  → " + new File(outGt).getName());
    }

    // Dirty ER conversion
    /**
     * For dirty ER (deduplication) the same profile collection is used as both
     * D1 and D2 so the blocking code can find within-collection duplicates.
     * JedAI's BilateralDuplicatePropagation handles this correctly when given
     * the same collection twice.
     */
    private static void convertDirtyER(Dataset ds) throws Exception {
        System.out.println("  Mode: dirty ER (deduplication)");

        List<EntityProfile> profiles = readProfiles(RAW_DIR + ds.fileA);
        System.out.printf("  Profiles: %,d%n", profiles.size());

        Map<String, Integer> idx = buildIndexMap(profiles);
        Set<IdDuplicates> gt = readGroundTruth(RAW_DIR + ds.gtFile, idx, idx);
        System.out.printf("  GT pairs: %,d%n", gt.size());

        // Write the same collection as both D1 and D2
        String outA  = OUT_DIR + ds.name + "ProfilesD1";
        String outB  = OUT_DIR + ds.name + "ProfilesD2";
        String outGt = OUT_DIR + ds.name + "Duplicates";

        writeProfiles(profiles, outA);
        writeProfiles(profiles, outB);
        writeGroundTruth(gt, outGt);

        System.out.println("  → " + new File(outA).getName());
        System.out.println("  → " + new File(outB).getName());
        System.out.println("  → " + new File(outGt).getName());
    }

    // CSV → EntityProfile list
    /**
     * Reads a CSV (with header row) and converts each row to an EntityProfile.
     *
     * - The 'id' column (first column, or any column named 'id', 'Id', 'rec_id')
     *   becomes the entity URL (the unique identifier).
     * - Every other non-empty column becomes an Attribute on the profile.
     * - Column names are used as attribute names so schema-based blocking
     *   works correctly in addition to schema-agnostic blocking.
     */
    private static List<EntityProfile> readProfiles(String csvPath) throws Exception {
        List<EntityProfile> profiles = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(csvPath), StandardCharsets.UTF_8))) {

            String headerLine = br.readLine();
            if (headerLine == null) return profiles;

            String[] headers = parseCsvLine(headerLine);

            // Find the id column (first column wins; fallback to column named id/Id/rec_id)
            int idCol = 0;
            for (int i = 0; i < headers.length; i++) {
                String h = headers[i].trim().toLowerCase();
                if (h.equals("id") || h.equals("rec_id") || h.equals("rxaui")) {
                    idCol = i;
                    break;
                }
            }

            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                String[] values = parseCsvLine(line);
                if (values.length == 0) continue;

                // Entity URL = the id column value
                String entityUrl = idCol < values.length ? values[idCol].trim() : line;
                EntityProfile profile = new EntityProfile(entityUrl);

                // Add all other non-empty columns as attributes
                for (int i = 0; i < headers.length; i++) {
                    if (i == idCol) continue;
                    if (i >= values.length) continue;
                    String val = values[i].trim();
                    if (!val.isEmpty() && !val.equals("\"\"")) {
                        profile.addAttribute(headers[i].trim(), val);
                    }
                }

                profiles.add(profile);
            }
        }
        return profiles;
    }

    // Ground truth CSV → IdDuplicates set
    /**
     * Reads a ground-truth CSV with columns id1, id2.
     * Looks up each id in the provided index maps and records the
     * (list-position-A, list-position-B) pair as an IdDuplicates entry.
     * Rows where either id is missing from the index are silently skipped.
     */
    private static Set<IdDuplicates> readGroundTruth(
            String gtPath,
            Map<String, Integer> idxA,
            Map<String, Integer> idxB) throws Exception {

        Set<IdDuplicates> gt = new HashSet<>();
        int missing = 0;

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(gtPath), StandardCharsets.UTF_8))) {

            String header = br.readLine(); // skip header
            if (header == null) return gt;

            // Find id1/id2 column positions
            String[] headers = parseCsvLine(header);
            int col1 = 0, col2 = 1;
            for (int i = 0; i < headers.length; i++) {
                if (headers[i].trim().equalsIgnoreCase("id1")) col1 = i;
                if (headers[i].trim().equalsIgnoreCase("id2")) col2 = i;
            }

            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = parseCsvLine(line);
                if (parts.length < 2) continue;

                String id1 = col1 < parts.length ? parts[col1].trim() : "";
                String id2 = col2 < parts.length ? parts[col2].trim() : "";

                Integer posA = idxA.get(id1);
                Integer posB = idxB.get(id2);

                if (posA == null || posB == null) {
                    missing++;
                    continue;
                }
                gt.add(new IdDuplicates(posA, posB));
            }
        }

        if (missing > 0) {
            System.out.printf("  Warning: %,d GT pairs skipped (ids not found in profiles)%n", missing);
        }
        return gt;
    }

    // Serialization
    @SuppressWarnings("unchecked")
    private static void writeProfiles(List<EntityProfile> profiles, String outPath) throws Exception {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(outPath)))) {
            oos.writeObject(profiles);
        }
    }

    @SuppressWarnings("unchecked")
    private static void writeGroundTruth(Set<IdDuplicates> gt, String outPath) throws Exception {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(outPath)))) {
            oos.writeObject(gt);
        }
    }

    // Helpers
    /** Build entity-URL → list-index map for ground-truth resolution. */
    private static Map<String, Integer> buildIndexMap(List<EntityProfile> profiles) {
        Map<String, Integer> map = new HashMap<>(profiles.size() * 2);
        for (int i = 0; i < profiles.size(); i++) {
            map.put(profiles.get(i).getEntityUrl(), i);
        }
        return map;
    }

    /**
     * Minimal CSV line parser that handles double-quoted fields with embedded
     * commas and escaped quotes ("").  Sufficient for all medical datasets used
     * here (none use multi-line fields).
     */
    private static String[] parseCsvLine(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder sb    = new StringBuilder();
        boolean inQuotes    = false;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        sb.append('"');
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    sb.append(c);
                }
            } else {
                if (c == '"') {
                    inQuotes = true;
                } else if (c == ',') {
                    fields.add(sb.toString());
                    sb.setLength(0);
                } else {
                    sb.append(c);
                }
            }
        }
        fields.add(sb.toString());
        return fields.toArray(new String[0]);
    }

    // Verification
    /** Quick sanity-check: re-read the serialized febrl1 files via JedAI readers. */
    private static void verifySerialized(String datasetName) {
        String pathD1 = OUT_DIR + datasetName + "ProfilesD1";
        String pathGt = OUT_DIR + datasetName + "Duplicates";

        File fD1 = new File(pathD1);
        File fGt = new File(pathGt);

        if (!fD1.exists() || !fGt.exists()) {
            System.out.println("  (skipped – files not found)");
            return;
        }

        try {
            IEntityReader reader = new EntitySerializationReader(pathD1);
            List<EntityProfile> profiles = reader.getEntityProfiles();
            System.out.printf("  D1 profiles re-read: %,d%n", profiles.size());
            if (!profiles.isEmpty()) {
                EntityProfile p = profiles.get(0);
                System.out.printf("  First entity URL: %s  |  Attributes: %d%n",
                    p.getEntityUrl(), p.getAttributes().size());
            }
        } catch (Exception e) {
            System.out.println("  Verification error: " + e.getMessage());
        }
    }
}
