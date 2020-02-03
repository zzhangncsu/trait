/*
/* Zhe Zhang and Munindar P. Singh. 2019.
/* Leveraging Structural and Semantic Correspondence for Attribute-Oriented Aspect Sentiment Discovery.
/* In Proceedings of the 24th Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1‐10, Hong Kong.
*/

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class TraitModel {

    private static int BURN_IN = 50;
    private static int THIN_INTERVAL = 50;
    private static int MAX_ITERATION = 500;
    private static int MAX_TEST_ITERATION = 20;
    private static boolean GENERATE_MODELS = true;
    private static int TOPN_WORD = 30;
    private static int ASPECTNUM = 20;
    private static int SENTINUM = 2;
    private static int TERMNUM = -1;
    private static final double NUMLEXICON = 2;

    private static final int POSITIVE = 0;
    private static final int NEGATIVE = 1;
    private static double BETA_POSITIVE = 5d;
    private static double BETA_NEGATIVE = 5d;
    private static double GAMMA = 50d / ASPECTNUM;

    private static double ALPHAGENERAL = .05d;
    private static double ALPHALEXICON = 5d;
    private static double ALPHANONLEXICON = 0d;
    private String stage = "Burn-in";
    private int updateNum;

    /***********************************
     * Arch for TripAdvisor cross
     ***********************************/

	private static double WORD_PROMOTION = 0.3;
	private static double LAMBDA = 1;
	private static double EPSILON = 0.6;
	private static String FOLD = "0";
	private static final String DATA = "resources/tripadvisor/cross_validation/fold_" + FOLD + "/train/";
	private static final String TERMINDEX = "resources/tripadvisor/indexWord/termIdx.txt";
	private static final String QUERYDATA = "resources/tripadvisor/cross_validation/fold_" + FOLD + "/test/";
	private static final String SIMILARITY = "resources/tripadvisor/cross_validation/fold_" + FOLD + "/similarity/" + "similarity.txt";
	private static final String SENSIMILARITY = "resources/tripadvisor/cross_validation/fold_" + FOLD + "/simTrain/";

	private static final String OUTPUTMODEL = "output/model/";
	private final String OUTPUTTOPWORDS = "output/topWordsTrip_" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION+"_" + LAMBDA
			+ ".txt";
	private static final String LEX = "resources/libs/SentiWords-";
	private final String LOGSENTI = "output/trip_all" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION + WORD_PROMOTION+"_" + LAMBDA + ".txt";

    /***********************************
     * Arch for Yelp cross
     ***********************************/
//    private static double WORD_PROMOTION = 0.1;
//    private static double LAMBDA = 1;
//    private static double EPSILON = 0.6;
//    private static String FOLD = "0";
//    private static final String DATA = "resources/yelp/cross_validation/fold_" + FOLD + "/train/";
//    private static final String TERMINDEX = "resources/yelp/indexWord/termIdx.txt";
//    private static final String QUERYDATA = "resources/yelp/cross_validation/fold_" + FOLD + "/test/";
//    private static final String SIMILARITY = "resources/yelp/cross_validation/fold_" + FOLD + "/similarity/" + "similarity.txt";
//	private static final String SENSIMILARITY = "resources/yelp/cross_validation/fold_" + FOLD + "/simTrain/";
//
//	private static final String OUTPUTMODEL = "output/model/";
//	private static final String OUTPUTTOPWORDS = "output/topWordsYelp_" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION+"_" + LAMBDA
//			+ ".txt";
//	private static final String LEX = "resources/libs/SentiWords-";
//	private static final String LOGSENTI = "output/yelp_all_" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION+"_" + LAMBDA
//			+ ".txt";

    /***********************************
     * Arch for Location
     ***********************************/
//    private static double WORD_PROMOTION = 0.1;
//    private static double LAMBDA = 1;
//    private static double EPSILON = 0.8;
//    private static String FOLD = "LOCATION";
//
//    private static final String DATA = "resources/tripadvisor/location/locationDataIndexed/";
//    private static final String TERMINDEX = "resources/location/indexWord/termIdx.txt";
//    private static final String QUERYDATA = "";
//    private static final String SIMILARITY = "resources/location/similarity/similarity_location_model.txt";
//    private static final String SENSIMILARITY = "resources/location/simTrain";
//
//    private static final String OUTPUTMODEL = "output/model/";
//    private final String OUTPUTTOPWORDS = "output/topWordsTrip_" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION + "_" + LAMBDA
//            + ".txt";
//    private static final String LEX = "resources/libs/SentiWords-";
//    private final String LOGSENTI = "output/trip_all" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION + "_" + LAMBDA + ".txt";

    /***********************************
     * Arch for Purpose
     ***********************************/
//	private static double WORD_PROMOTION = 0.1;
//	private static double LAMBDA = 1;
//	private static double EPSILON = 0.8;
//	private static String FOLD = "PURPOSE";
//
//	private static final String DATA = "resources/type//purposeDataIndexed/";
//	private static final String TERMINDEX = "resources/type/indexWord/termIdx.txt";
//	private static final String QUERYDATA = "";
//	private static final String SIMILARITY = "resources/type/similarity/similarity_purpose_model.txt";
//	private static final String SENSIMILARITY = "resources/type/simTrain";
//
//	private static final String OUTPUTMODEL = "output/model/";
//	private final String OUTPUTTOPWORDS = "output/model/topWordsTrip_" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION + "_" + LAMBDA
//			+ ".txt";
//	private static final String LEX = "resources/libs/SentiWords-";
//	private final String LOGSENTI = "output/trip_all" + ASPECTNUM + "_" + FOLD + "_" + WORD_PROMOTION + "_" + LAMBDA + ".txt";

    private ArrayList<String> filePathList = null;
    private HashMap<String, Integer> userIdxMap = null;
    private HashMap<Integer, String> idxUserMap = null;
    private HashMap<Integer, Word> idxWordMap = null;

    private HashMap<String, Integer> termIdxMap = null;
    private HashMap<Integer, String> idxTermMap = null;

    private ArrayList<ArrayList<String>> lexList = null;
    private ArrayList<HashSet<String>> sentiWordCountSetList = null;

    private double[] beta = null;
    private double[] gamma = null;
    private double[] alphaAll = null;

    private ArrayList<int[]> ctDocSenti = null;
    private ArrayList<Integer> ctDocSentiSum = null;
    private ArrayList<int[][]> ctUserSentiAspect = null;
    private ArrayList<int[]> ctUserSentiAspectSum = null;
    private ArrayList<double[][]> ctSentiAspectTerm = null;
    private ArrayList<double[]> ctSentiAspectTermSum = null;
    private ArrayList<Sentence[][]> szUserDoc = null;
    private ArrayList<Sentence[][]> szUserDocQuery = null;
    private ArrayList<Sentence[][]> szUserDocQueryOri = null;
    private ArrayList<ArrayList<ArrayList<ProbEle>>> phiMatrix = null;
    private ArrayList<ArrayList<ArrayList<ProbEle>>> phiMatrixRel = null;
    private ArrayList<ArrayList<ArrayList<ProbEle>>> piMatrix = null;
    private ArrayList<ArrayList<ArrayList<ProbEle>>> piMatrixRel = null;
    private ArrayList<ArrayList<ProbEle>> psiMatrix = null;
    private HashMap<Integer, ArrayList<SemanticPair>> semanticMap = null;
    private ArrayList<int[]> ctDocSentiOri = null;
    private ArrayList<Integer> ctDocSentiSumOri = null;
    private ArrayList<int[][]> ctUserSentiAspectOri = null;
    private ArrayList<int[]> ctUserSentiAspectSumOri = null;
    private ArrayList<double[][]> ctSentiAspectTermOri = null;
    private ArrayList<double[]> ctSentiAspectTermSumOri = null;
    private HashMap<Integer, Integer> thresholdMapTesting = null;
    private HashMap<Integer, Integer> thresholdMapTraining = null;
    private HashMap<Integer, HashMap<Integer, String>> idxMap = null;
    private Random generator = null;
    private static Logger logger = null;
    private ArrayList<ArrayList<String>> szUserDocRate = null;
    private HashMap<Integer, HashMap<String, List<String>>> sentenceMap = null;
    private HashMap<String, Sentence> sentMap = null;

    public TraitModel(int aspectNum, double beta1, double beta2, double gamma, double alphaGeneral,
                      double alphaLexicon, double alphaNonlexicon, double lambda, double rho, double epsilon, int maxIter, int maxIterTesting, int burnIn, int thinInterval,
                      boolean outputModel) {
        TraitModel.ASPECTNUM = aspectNum;
        TraitModel.BETA_POSITIVE = beta1;
        TraitModel.BETA_NEGATIVE = beta2;
        TraitModel.GAMMA = gamma;
        TraitModel.ALPHAGENERAL = alphaGeneral;
        TraitModel.ALPHALEXICON = alphaLexicon;
        TraitModel.ALPHANONLEXICON = alphaNonlexicon;
        TraitModel.LAMBDA = lambda;
        TraitModel.WORD_PROMOTION = rho;
        TraitModel.EPSILON = epsilon;
        TraitModel.BURN_IN = burnIn;
        TraitModel.THIN_INTERVAL = thinInterval;
        TraitModel.MAX_ITERATION = maxIter;
        TraitModel.MAX_TEST_ITERATION = maxIterTesting;
        TraitModel.GENERATE_MODELS = outputModel;

        logger = LogManager.getLogger(TraitModel.class.getName());

        System.out.println("\n/********************************/");
        System.out.println("/* Number of aspects: " + ASPECTNUM);
        System.out.println("/* Positive beta: " + BETA_POSITIVE);
        System.out.println("/* Negative beta: " + BETA_NEGATIVE);
        System.out.println("/* Gamma: " + GAMMA);
        System.out.println("/* Alpha general: " + ALPHAGENERAL);
        System.out.println("/* Alpha lexicon: " + ALPHALEXICON);
        System.out.println("/* Alpha non-lexicon: " + ALPHANONLEXICON);
        System.out.println("/* Word promotion: " + WORD_PROMOTION);
        System.out.println("/* Word similarity threshold: " + EPSILON);
        System.out.println("/* Discourse promotion: " + LAMBDA);
        System.out.println("/* Output Model: " + GENERATE_MODELS);
        System.out.println("/* Max iterations: " + MAX_ITERATION);
        System.out.println("/* Burn-in iterations: " + BURN_IN);
        System.out.println("/* Samling interval: " + THIN_INTERVAL);
        System.out.println("/* Testing iterations: " + MAX_TEST_ITERATION);
        System.out.println("/********************************/\n");
    }

    private void readSentenceSimilarity() {
        try {
            System.out.println("Read sentence similarity map...");
            logger.log(Level.DEBUG, "Read sentence similarity START");

            sentenceMap = new HashMap<>();
            File dir = new File(SENSIMILARITY);
            File[] files = dir.listFiles();
            assert files != null;
            for (File f : files) {
                if (f.isFile()) {
                    if (f.getPath().contains(".DS_Store")) {
                        continue;
                    }
                    int idxLastSlash = f.getPath().lastIndexOf("\\");
                    String username = f.getPath().substring(idxLastSlash + 1);
                    int userId = userIdxMap.get(username);
                    HashMap<String, List<String>> userSenMap = new HashMap<>();
                    BufferedReader reader = new BufferedReader(new FileReader(f.getPath()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] tokens = line.split("#");
                        String senId = tokens[0];
                        List<String> simSenList = Arrays.asList(tokens[1].split(","));
                        userSenMap.put(senId, simSenList);
                    }
                    reader.close();
                    sentenceMap.put(userId, userSenMap);
                }
            }
            logger.log(Level.DEBUG, "Read sentence similarity END");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    private void readSemanticMapping() {
        try {
            System.out.println("Read word similarity list...");
            logger.log(Level.DEBUG, "Read word similarity list START");
            semanticMap = new HashMap<>();
            BufferedReader reader = new BufferedReader(new FileReader(SIMILARITY));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                String word = tokens[0];
                if (termIdxMap.containsKey(word)) {
                    int wordIdx = termIdxMap.get(word);
                    ArrayList<SemanticPair> similarList;
                    if (semanticMap.containsKey(wordIdx)) {
                        similarList = semanticMap.get(wordIdx);
                    } else {
                        similarList = new ArrayList<>();
                        semanticMap.put(wordIdx, similarList);
                    }
                    for (int i = 1; i < tokens.length; i++) {
                        String[] subTokens = tokens[i].split("\\|");
                        String key = subTokens[0].toLowerCase();
                        if (key.trim().length() == 0) {
                            continue;
                        }
                        if (termIdxMap.containsKey(key)) {
                            int keyIdx = termIdxMap.get(key);
                            double similarity = Double.parseDouble(subTokens[1]);
                            double posSimilarity = Double.parseDouble(subTokens[2]);
                            double negSimilarity = Double.parseDouble(subTokens[3]);
                            if (!word.trim().equals(key.trim())) {
                                SemanticPair pair = new SemanticPair(keyIdx, similarity, posSimilarity, negSimilarity);
                                similarList.add(pair);
                            }
                        }
                    }
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void readTermMapping() {
        try {
            System.out.println("Read term mapping list...");
            logger.log(Level.DEBUG, "Read term mapping list START");
            BufferedReader reader = new BufferedReader(new FileReader(TERMINDEX));
            idxTermMap = new HashMap<>();
            termIdxMap = new HashMap<>();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.trim().split("\t");
                if (!idxTermMap.containsKey(Integer.valueOf(tokens[1]))) {
                    idxTermMap.put(Integer.valueOf(tokens[1]), tokens[0]);
                }
                if (!termIdxMap.containsKey(tokens[0])) {
                    termIdxMap.put(tokens[0], Integer.valueOf(tokens[1]));
                }
            }
            TERMNUM = idxTermMap.size();
            reader.close();
            logger.log(Level.DEBUG, "Read term mapping list END");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void readLex() {
        try {
            System.out.println("Read sentiment word list...");
            logger.log(Level.DEBUG, "Read sentiment word list START");
            lexList = new ArrayList<>();
            for (int s = 0; s < NUMLEXICON; s++) {
                BufferedReader reader = new BufferedReader(new FileReader(LEX + s));
                ArrayList<String> lex = new ArrayList<>();
                String line;
                while ((line = reader.readLine()) != null) {
                    lex.add(line.trim());
                }
                reader.close();
                lexList.add(lex);
            }
            logger.log(Level.DEBUG, "Read sentiment word list END");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void getFilePathList() {
        logger.log(Level.DEBUG, "Read training file list START");
        filePathList = new ArrayList<>();
        userIdxMap = new HashMap<>();
        idxUserMap = new HashMap<>();
        File dir = new File(DATA);
        File[] files = dir.listFiles();
        int count = 0;
        assert files != null;
        for (File f : files) {
            if (f.isFile()) {
                if (f.getPath().contains(".DS_Store")) {
                    continue;
                }
                filePathList.add(f.getPath());
                int idxLastSlash = f.getPath().lastIndexOf("\\");
                String username = f.getPath().substring(idxLastSlash + 1);
                if (!userIdxMap.containsKey(username)) {
                    userIdxMap.put(username, count);
                    idxUserMap.put(count, username);
                }
                count++;
            }
        }
        logger.log(Level.DEBUG, "Read training file list END");
    }

    private void getResources() {
        getFilePathList();
        readSentenceSimilarity();

        readTermMapping();
        readLex();
        generator = new Random(System.currentTimeMillis());
        readSemanticMapping();
    }

    private void initialize() {
        try {
            logger.log(Level.DEBUG, "Initialization START");
            thresholdMapTraining = new HashMap<>();

            phiMatrix = new ArrayList<>();
            phiMatrixRel = new ArrayList<>();

            piMatrix = new ArrayList<>();
            piMatrixRel = new ArrayList<>();

            psiMatrix = new ArrayList<>();

            ctDocSenti = new ArrayList<>();
            ctDocSentiSum = new ArrayList<>();

            ctUserSentiAspect = new ArrayList<>();
            ctUserSentiAspectSum = new ArrayList<>();
            ctSentiAspectTerm = new ArrayList<>();
            ctSentiAspectTermSum = new ArrayList<>();

            idxWordMap = new HashMap<>();
            szUserDoc = new ArrayList<>();

            double[] alpha = new double[TERMNUM];
            beta = new double[SENTINUM];
            gamma = new double[ASPECTNUM];

            alphaAll = new double[SENTINUM];
            Arrays.fill(alpha, ALPHAGENERAL);
            Arrays.fill(gamma, GAMMA);

            beta[POSITIVE] = BETA_POSITIVE;
            beta[NEGATIVE] = BETA_NEGATIVE;

            sentiWordCountSetList = new ArrayList<>();
            sentMap = new HashMap<>();
            szUserDocRate = new ArrayList<>();
            idxMap = new HashMap<>();

            for (int i = 0; i < SENTINUM; i++) {
                double[][] ctAspectTerm = new double[ASPECTNUM][TERMNUM];
                ctSentiAspectTerm.add(ctAspectTerm);

                double[] ctSentiAspectSum = new double[ASPECTNUM];
                ctSentiAspectTermSum.add(ctSentiAspectSum);

                HashSet<String> sentiWordCountSet = new HashSet<>();
                sentiWordCountSetList.add(sentiWordCountSet);
            }

            for (String filePath : filePathList) {
                String line;
                BufferedReader reader = new BufferedReader(new FileReader(filePath));
                System.out.println(filePath);
                int idxLastSlash = filePath.lastIndexOf("\\");
                String username = filePath.substring(idxLastSlash + 1);
                int userIdx = userIdxMap.get(username);
                Sentence[][] szDoc = new Sentence[100000][];

                int[][] ctSentiAspect = new int[SENTINUM][ASPECTNUM];
                int[] ctDocAspectSum = new int[SENTINUM];
                ArrayList<String> currentUserRate = new ArrayList<>();
                HashMap<Integer, String> idxMapUser = new HashMap<>();

                int docIdx = 0;

                while ((line = reader.readLine()) != null) {
                    if (line.contains("[IDX]:")) {
                        int numSentence = -1;
                        Pattern pIdx = Pattern
                                .compile("\\[IDX]:(.*?)\\|\\[ID]:(.*?)\\|\\[SEN]:(.*?)\\|\\[R]:(.*?)\\|(.*?)");
                        Matcher mIdx = pIdx.matcher(line);
                        double rate = -1d;
                        String rId = "";

                        if (mIdx.matches()) {
                            rId = mIdx.group(2);
                            numSentence = Integer.parseInt(mIdx.group(3));
                            rate = Double.parseDouble(mIdx.group(4));
                        }

                        idxMapUser.put(docIdx, rId);

                        String sRate = "";

                        if (rate < 3) {
                            sRate = "neg";
                        } else if (rate >= 3) {
                            sRate = "pos";
                        } else {
                            System.out.println("negative rating!");
                        }

                        currentUserRate.add(sRate);

                        szDoc[docIdx] = new Sentence[numSentence];

                        int[] ctSenti = new int[SENTINUM];
                        int ctSentiSum = 0;

                        int sentIdx = 0;
                        while ((line = reader.readLine()).trim().length() != 0) {
                            String[] tokens = line.split("\\|");
                            line = tokens[0];

                            Pattern sIdx = Pattern.compile("\\[(.*?)]\\[(.*?)](.*?)");
                            Matcher msIdx = sIdx.matcher(line);
                            int oriSId = -1;

                            if (msIdx.matches()) {
                                oriSId = Integer.parseInt(msIdx.group(2));
                            }

                            line = line.replaceAll("\\[[0-9]+]\\[[0-9]+]", "");
                            double rAspect = generator.nextDouble() * ASPECTNUM;
                            double rSentiment = generator.nextDouble() * SENTINUM;
                            int aspect = (int) rAspect;
                            int sentiment = (int) rSentiment;

                            ArrayList<Word> wordList = new ArrayList<>();
                            String[] words = line.split("[\\s]+");

                            int sentiLex = -1;
                            boolean setSentiLex = false;
                            for (String word : words) {
                                String curWord = idxTermMap.get(Integer.valueOf(word));
                                int curWordIdx = termIdxMap.get(curWord);
                                Word w = new Word();
                                w.setTermIdx(curWordIdx);

                                for (int l = 0; l < lexList.size(); l++) {
                                    if (setSentiLex) {
                                        setSentiLex = false;
                                        break;
                                    }
                                    ArrayList<String> lexicon = lexList.get(l);
                                    for (String lex : lexicon) {
                                        if (curWord.equalsIgnoreCase(lex)) {
                                            w.setSentiLex(l);
                                            sentiLex = l;
                                            setSentiLex = true;

                                            HashSet<String> sentiWordCountSet = sentiWordCountSetList.get(l);
                                            if (!sentiWordCountSet.add(lex)) {
                                                sentiWordCountSet.add(lex);
                                            }
                                            break;
                                        }
                                    }
                                }
                                wordList.add(w);
                                if (!idxWordMap.containsKey(w.getTermIdx())) {
                                    idxWordMap.put(w.getTermIdx(), w);
                                }
                            }

                            if (sentiLex >= 0) {
                                if (sentiment != sentiLex) {
                                    sentiment = sentiLex;
                                }
                            }

                            ctSenti[sentiment]++;
                            ctSentiSum++;
                            ctSentiAspect[sentiment][aspect]++;
                            ctDocAspectSum[sentiment]++;
                            double[][] ctAspectTerm = ctSentiAspectTerm.get(sentiment);
                            double[] ctSentiAspectSum = ctSentiAspectTermSum.get(sentiment);

                            for (Word w : wordList) {
                                int curWordIdx = w.getTermIdx();

                                w.setAspect(aspect);
                                w.setSentiment(sentiment);
                                ctAspectTerm[aspect][curWordIdx]++;
                                ctSentiAspectSum[aspect]++;
                            }

                            Sentence sent = new Sentence(oriSId, rId, sentiment, aspect, wordList);
                            if (sentMap.containsKey(rId + "|" + oriSId)) {
                                System.err.println("Error RID");
                            } else {
                                sentMap.put(rId + "|" + oriSId, sent);
                            }

                            szDoc[docIdx][sentIdx] = sent;
                            sentIdx++;
                        }
                        ctDocSenti.add(ctSenti);
                        ctDocSentiSum.add(ctSentiSum);
                        docIdx++;
                    }
                }

                ctUserSentiAspect.add(ctSentiAspect);
                ctUserSentiAspectSum.add(ctDocAspectSum);
                szUserDocRate.add(currentUserRate);
                idxMap.put(userIdx, idxMapUser);

                reader.close();
                szUserDoc.add(szDoc);
            }

            System.out.println("end");

            int sentiWordCountAll = 0;
            for (int s = 0; s < SENTINUM; s++) {
                sentiWordCountAll += sentiWordCountSetList.get(s).size();
            }
            int numGeneral = idxTermMap.size() - sentiWordCountAll;
            for (int s = 0; s < SENTINUM; s++) {
                alphaAll[s] = numGeneral * ALPHAGENERAL + sentiWordCountSetList.get(s).size() * ALPHALEXICON
                        + (sentiWordCountAll - sentiWordCountSetList.get(s).size()) * ALPHANONLEXICON;
            }
            logger.log(Level.DEBUG, "Initialization END");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void gibbsSampling() {
        logger.log(Level.DEBUG, "Sampling START");

        int docThreshold = 0;
        for (String filePath : filePathList) {
            int idxLastSlash = filePath.lastIndexOf("\\");
            String username = filePath.substring(idxLastSlash + 1);
            if (username.contains(".DS_Store")) {
                continue;
            }

            int userIdx = userIdxMap.get(username);
            Sentence[][] szDoc = szUserDoc.get(userIdx);

            int docCount = 0;

            thresholdMapTraining.put(userIdx, docThreshold);

            for (int docIdx = 0; docIdx < szDoc.length; docIdx++) {

                if (szDoc[docIdx] == null) {
                    break;
                }

                int[] ctSenti = ctDocSenti.get(docThreshold + docIdx);
                Integer ctSentiSum = ctDocSentiSum.get(docThreshold + docIdx);
                for (int sentenceIdx = 0; sentenceIdx < szDoc[docIdx].length; sentenceIdx++) {

                    Sentence sentence = szDoc[docIdx][sentenceIdx];

                    int aspect = sentence.getAspect();
                    int sentiment = sentence.getSentiment();

                    List<String> relatedSentenceIds = sentenceMap.get(userIdx).get(sentence.getrIdx() + "|" + sentence.getOriIdx());
                    List<Sentence> relatedSentences = new ArrayList<>();

                    if (relatedSentenceIds != null) {
                        for (String id : relatedSentenceIds) {
                            if (sentMap.containsKey(id)) {
                                relatedSentences.add(sentMap.get(id));
                            }
                        }
                    }

                    ctSenti[sentiment]--;
                    ctSentiSum--;
                    ctDocSentiSum.set(docThreshold + docIdx, ctSentiSum);

                    int[][] ctSentiAspect = ctUserSentiAspect.get(userIdx);
                    ctSentiAspect[sentiment][aspect]--;
                    int[] ctDocAspectSum = ctUserSentiAspectSum.get(userIdx);
                    ctDocAspectSum[sentiment]--;

                    double[][] ctAspectTerm = ctSentiAspectTerm.get(sentiment);
                    double[] ctSentiAspectSum = ctSentiAspectTermSum.get(sentiment);
                    for (Word w : sentence.getWordList()) {
                        int curWordIdx = w.getTermIdx();
                        double reducedCount = ctAspectTerm[aspect][curWordIdx] - 1;
                        ctAspectTerm[aspect][curWordIdx] = Math.max(0, reducedCount);

                        ctSentiAspectSum[aspect]--;
                        if (semanticMap.containsKey(curWordIdx)) {
                            if (stage.equals("Regular")) {
                                ArrayList<Integer> promotionList = w.getPromotionList();
                                if (promotionList != null) {
                                    for (int similarIdx : promotionList) {
                                        double reduced = ctAspectTerm[aspect][similarIdx] - WORD_PROMOTION;
                                        ctAspectTerm[aspect][similarIdx] = Math.max(0, reduced);
                                        double reduced_all = ctSentiAspectSum[aspect] - WORD_PROMOTION;
                                        ctSentiAspectSum[aspect] = Math.max(0, reduced_all);
                                    }
                                }
                            }
                        }

                    }

                    int sentiLex = -1;
                    for (Word w : sentence.getWordList()) {
                        if (w.getSentiLex() >= 0) {
                            sentiLex = w.getSentiLex();
                            break;
                        }
                    }

                    double[][] probSentiTopic = new double[SENTINUM][ASPECTNUM];
                    double probSum = 0d;

                    HashSet<Word> wordSet = sentence.getWordSet();

                    for (int s = 0; s < SENTINUM; s++) {
                        ctAspectTerm = ctSentiAspectTerm.get(s);
                        ctSentiAspectSum = ctSentiAspectTermSum.get(s);

                        for (int a = 0; a < ASPECTNUM; a++) {

                            if (sentiLex >= 0) {
                                if (s != sentiLex) {
                                    probSentiTopic[s][a] = 0d;
                                    continue;
                                }
                            }

                            double part1 = (ctSentiAspect[s][a] + gamma[a])
                                    / (ctDocAspectSum[s] + ASPECTNUM * gamma[a]);
                            double part2 = (ctSenti[s] + beta[s]) / (ctSentiSum + beta[POSITIVE] + beta[NEGATIVE]);

                            Iterator<Word> iterWord = wordSet.iterator();
                            double part3 = 1d;

                            int allCount = 0;
                            double aspectSum = ctSentiAspectSum[a];
                            double alphaSum = alphaAll[s];

                            while (iterWord.hasNext()) {
                                Word w = iterWord.next();
                                int curWordIdx = w.getTermIdx();

                                int wordCount = sentence.getWordCount(curWordIdx);
                                double alpha;
                                if (w.getSentiLex() < 0) {
                                    alpha = ALPHAGENERAL;
                                } else {
                                    if (w.getSentiLex() == s) {
                                        alpha = ALPHALEXICON;
                                    } else {
                                        alpha = ALPHANONLEXICON;
                                    }
                                }

                                for (int c = 0; c < wordCount; c++) {
                                    part3 *= (ctAspectTerm[a][curWordIdx] + alpha + c)
                                            / (aspectSum + alphaSum + allCount);
                                    allCount++;
                                }
                            }

                            double mrfWeight = 0;
                            double lambda = LAMBDA;

                            if (relatedSentences.size() > 0) {
                                for (Sentence sent : relatedSentences) {
                                    if (sent.getAspect() == a) {
                                        mrfWeight++;
                                    }
                                }

                                mrfWeight /= relatedSentences.size();
                            }

                            probSentiTopic[s][a] = part1 * part2 * part3 * Math.exp(lambda * mrfWeight);
                            probSum += probSentiTopic[s][a];
                        }

                    }

                    double threshold = 0d;
                    double r = generator.nextDouble() * probSum;
                    boolean isFound = false;

                    int newAspect = -1;
                    int newSentiment = -1;

                    for (int s = 0; s < SENTINUM; s++) {
                        for (int a = 0; a < ASPECTNUM; a++) {
                            threshold += probSentiTopic[s][a];
                            if (r <= threshold) {
                                newAspect = a;
                                newSentiment = s;
                                isFound = true;
                                break;
                            }
                        }
                        if (isFound) {
                            break;
                        }
                    }

                    sentence.setAspect(newAspect);
                    sentence.setSentiment(newSentiment);
                    szDoc[docIdx][sentenceIdx] = sentence;

                    ctSenti[newSentiment]++;
                    ctSentiSum++;
                    ctDocSentiSum.set(docThreshold + docIdx, ctSentiSum);

                    ctSentiAspect[newSentiment][newAspect]++;
                    ctDocAspectSum[newSentiment]++;

                    ctAspectTerm = ctSentiAspectTerm.get(newSentiment);
                    ctSentiAspectSum = ctSentiAspectTermSum.get(newSentiment);
                    for (Word w : sentence.getWordList()) {
                        int curWordIdx = w.getTermIdx();
                        ctAspectTerm[newAspect][curWordIdx]++;
                        ctSentiAspectSum[newAspect]++;
                        if (semanticMap.containsKey(curWordIdx)) {
                            if (stage.equals("Regular")) {
                                ArrayList<Integer> promotionList = new ArrayList<>();
                                ArrayList<SemanticPair> similarList = semanticMap.get(curWordIdx);

                                for (int i = 0; i < similarList.size(); i++) {
                                    SemanticPair pair = similarList.get(i);
                                    int similarIdx = pair.getWordIdx();
                                    double similarity = pair.getSimilarity();
                                    if (newSentiment == 1 && pair.getPosSimilarity() >= pair.getNegSimilarity()) {
                                        continue;
                                    }

                                    if (newSentiment == 0 && pair.getPosSimilarity() < pair.getNegSimilarity()) {
                                        continue;
                                    }

                                    if (similarity > 0.6) {
                                        ctAspectTerm[newAspect][similarIdx] += WORD_PROMOTION;
                                        ctSentiAspectSum[newAspect] += WORD_PROMOTION;
                                        promotionList.add(similarIdx);
                                    }
                                    if (i == 4) {
                                        break;
                                    }
                                }
                                w.setPromotionList(promotionList);
                            }
                        }

                    }
                }
                docCount++;
            }
            docThreshold += docCount;
        }
        logger.log(Level.DEBUG, "Sampling END");
    }

    private void calculatePhiMatrix() {
        logger.log(Level.DEBUG, "Calculating Phi START");

        for (int s = 0; s < ctSentiAspectTerm.size(); s++) {
            ArrayList<ArrayList<ProbEle>> aspectTerm;
            if (phiMatrix.size() <= s) {
                aspectTerm = new ArrayList<>();
                phiMatrix.add(aspectTerm);
            } else {
                aspectTerm = phiMatrix.get(s);
            }

            double[][] ctAspectTerm = ctSentiAspectTerm.get(s);
            double[] ctSentiAspectSum = ctSentiAspectTermSum.get(s);

            for (int a = 0; a < ctAspectTerm.length; a++) {

                ArrayList<ProbEle> termList;
                if (aspectTerm.size() <= a) {
                    termList = new ArrayList<>();
                    aspectTerm.add(termList);
                } else {
                    termList = aspectTerm.get(a);
                }

                double aspectSum = ctSentiAspectSum[a];
                for (int t = 0; t < ctAspectTerm[a].length; t++) {
                    Word w = idxWordMap.get(t);
                    double alpha;
                    if (w != null) {
                        if (w.getSentiLex() < 0) {
                            alpha = ALPHAGENERAL;
                        } else {
                            if (w.getSentiLex() == s) {
                                alpha = ALPHALEXICON;
                            } else {
                                alpha = ALPHANONLEXICON;
                            }
                        }
                    } else {
                        alpha = ALPHAGENERAL;
                    }

                    double alphaSum = alphaAll[s];
                    double phi = (ctAspectTerm[a][t] + alpha) / (aspectSum + alphaSum);
                    ProbEle term;
                    if (termList.size() <= t) {
                        term = new ProbEle(t, phi);
                        termList.add(term);
                    } else {
                        term = termList.get(t);
                        double prob = term.getProb();
                        term.setProb(phi + prob);
                    }

                }
            }
        }
        logger.log(Level.DEBUG, "Calculating Phi END");
    }

    private void calculateFinalPhiMatrix() {
        logger.log(Level.DEBUG, "Calculating final Phi START");

        for (int s = 0; s < ctSentiAspectTerm.size(); s++) {
            ArrayList<ArrayList<ProbEle>> aspectTerm = phiMatrix.get(s);
            ArrayList<ArrayList<ProbEle>> aspectTermRel = new ArrayList<>();

            double[][] ctAspectTerm = ctSentiAspectTerm.get(s);

            HashMap<Integer, Double> termProbSumMap = new HashMap<>();
            HashMap<Integer, Double> termProbSumNonZeroCountMap = new HashMap<>();

            for (int a = 0; a < ctAspectTerm.length; a++) {
                ArrayList<ProbEle> termList = aspectTerm.get(a);
                for (int t = 0; t < ctAspectTerm[a].length; t++) {
                    ProbEle term = termList.get(t);
                    double phi = term.getProb() / updateNum;
                    term.setProb(phi);
                    if (phi > 0d) {
                        if (termProbSumMap.containsKey(t)) {
                            double sum = termProbSumMap.get(t);
                            sum *= phi;
                            termProbSumMap.put(t, sum);
                        } else {
                            termProbSumMap.put(t, phi);
                        }
                        if (termProbSumNonZeroCountMap.containsKey(t)) {
                            double count = termProbSumNonZeroCountMap.get(t);
                            termProbSumNonZeroCountMap.put(t, count + 1d);
                        } else {
                            termProbSumNonZeroCountMap.put(t, 1d);
                        }
                    }
                }
            }

            for (int a = 0; a < ASPECTNUM; a++) {
                ArrayList<ProbEle> termList = aspectTerm.get(a);
                ArrayList<ProbEle> relativeList = new ArrayList<>();
                for (int t = 0; t < TERMNUM; t++) {
                    int termIdx = termList.get(t).getIdx();
                    double phi = termList.get(t).getProb();
                    double relativePhi = Double.MIN_VALUE;
                    if (phi > 0d) {
                        double phiSum = termProbSumMap.get(termIdx);
                        if (phiSum == 0) {
                            phiSum = 0.000001d;
                        }
                        double nonZeroCount = termProbSumNonZeroCountMap.get(termIdx);
                        relativePhi = Math.pow(phiSum, (1d / nonZeroCount));
                        relativePhi = phi / relativePhi;
                        relativePhi = Math.log(relativePhi);
                        relativePhi = phi * relativePhi;
                    }
                    ProbEle term = new ProbEle(termIdx, relativePhi);
                    relativeList.add(term);
                }
                Collections.sort(relativeList);
                aspectTermRel.add(relativeList);
            }
            phiMatrixRel.add(aspectTermRel);

            for (int a = 0; a < ASPECTNUM; a++) {
                ArrayList<ProbEle> termList = aspectTerm.get(a);
                Collections.sort(termList);
            }
        }
        logger.log(Level.DEBUG, "Calculating Phi END");
    }

    private void calculatePiMatrix() {
        logger.log(Level.DEBUG, "Calculating Pi START");

        for (int u = 0; u < ctUserSentiAspect.size(); u++) {
            int[][] ctSentiAspect = ctUserSentiAspect.get(u);
            int[] ctDocAspectSum = ctUserSentiAspectSum.get(u);

            ArrayList<ArrayList<ProbEle>> sentiAspectList;
            if (piMatrix.size() <= u) {
                sentiAspectList = new ArrayList<>();
                piMatrix.add(sentiAspectList);
            } else {
                sentiAspectList = piMatrix.get(u);
            }

            for (int s = 0; s < SENTINUM; s++) {
                ArrayList<ProbEle> sentiList;
                if (sentiAspectList.size() <= s) {
                    sentiList = new ArrayList<>();
                    sentiAspectList.add(sentiList);
                } else {
                    sentiList = sentiAspectList.get(s);
                }
                int aspectSum = ctDocAspectSum[s];
                for (int a = 0; a < ASPECTNUM; a++) {
                    double pi = (ctSentiAspect[s][a] + gamma[a]) / (aspectSum + ASPECTNUM * gamma[a]);
                    ProbEle sentiAspect;
                    if (sentiList.size() <= a) {
                        sentiAspect = new ProbEle(s, a, pi);
                        sentiList.add(sentiAspect);
                    } else {
                        sentiAspect = sentiList.get(a);
                        double prob = sentiAspect.getProb();
                        sentiAspect.setProb(pi + prob);
                    }
                }
            }
        }

        logger.log(Level.DEBUG, "Calculating Pi END");
    }

    private void calculateFinalPiMatrix() {
        logger.log(Level.DEBUG, "Calculating final Pi START");

        HashMap<Integer, HashMap<Integer, Double>> aspectProbProdMapList = new HashMap<>();
        HashMap<Integer, HashMap<Integer, Double>> aspectProbSumNonZeroCountMapList = new HashMap<>();

        for (int u = 0; u < ctUserSentiAspect.size(); u++) {
            ArrayList<ArrayList<ProbEle>> sentiAspectList = piMatrix.get(u);

            for (int s = 0; s < SENTINUM; s++) {
                ArrayList<ProbEle> sentiList = sentiAspectList.get(s);

                HashMap<Integer, Double> aspectProbProdMap;
                HashMap<Integer, Double> aspectProbSumNonZeroCountMap;

                if (aspectProbProdMapList.containsKey(s)) {
                    aspectProbProdMap = aspectProbProdMapList.get(s);
                    aspectProbSumNonZeroCountMap = aspectProbSumNonZeroCountMapList.get(s);
                } else {
                    aspectProbProdMap = new HashMap<>();
                    aspectProbSumNonZeroCountMap = new HashMap<>();
                }

                for (int a = 0; a < ASPECTNUM; a++) {
                    ProbEle sentiAspect = sentiList.get(a);
                    double pi = sentiAspect.getProb() / updateNum;
                    sentiAspect.setProb(pi);

                    if (pi > 0d) {
                        if (aspectProbProdMap.containsKey(a)) {
                            double sum = aspectProbProdMap.get(a);
                            sum *= pi;
                            aspectProbProdMap.put(a, sum);
                        } else {
                            aspectProbProdMap.put(a, pi);
                        }
                        if (aspectProbSumNonZeroCountMap.containsKey(a)) {
                            double count = aspectProbSumNonZeroCountMap.get(a);
                            aspectProbSumNonZeroCountMap.put(a, count + 1d);
                        } else {
                            aspectProbSumNonZeroCountMap.put(a, 1d);
                        }
                    }

                }

                if (!aspectProbProdMapList.containsKey(s)) {
                    aspectProbProdMapList.put(s, aspectProbProdMap);
                    aspectProbSumNonZeroCountMapList.put(s, aspectProbSumNonZeroCountMap);
                }
            }
        }

        for (int u = 0; u < ctUserSentiAspect.size(); u++) {
            ArrayList<ArrayList<ProbEle>> sentiAspectList = piMatrix.get(u);
            ArrayList<ArrayList<ProbEle>> sentiAspectListRel = new ArrayList<>();

            for (int s = 0; s < SENTINUM; s++) {
                HashMap<Integer, Double> aspectProbProdMap = aspectProbProdMapList.get(s);
                HashMap<Integer, Double> aspectProbSumNonZeroCountMap = aspectProbSumNonZeroCountMapList.get(s);
                ArrayList<ProbEle> sentiList = sentiAspectList.get(s);
                ArrayList<ProbEle> relativeList = new ArrayList<>();
                for (int a = 0; a < ASPECTNUM; a++) {
                    double pi = sentiList.get(a).getProb();
                    double relativePi = Double.MIN_VALUE;
                    if (pi > 0d) {
                        double piSum = aspectProbProdMap.get(a);
                        double nonZeroCount = aspectProbSumNonZeroCountMap.get(a);
                        if (piSum == 0) {
                            piSum = 0.000001d;
                        }
                        relativePi = Math.pow(piSum, (1d / nonZeroCount));
                        relativePi = pi / relativePi;
                        relativePi = Math.log(relativePi);
                        relativePi = pi * relativePi;
                    }
                    if (relativePi < 0) {
                        relativePi = relativePi + 0d;
                    }
                    ProbEle aspect = new ProbEle(s, a, relativePi);
                    relativeList.add(aspect);
                }
                Collections.sort(relativeList);
                sentiAspectListRel.add(relativeList);
            }

            piMatrixRel.add(sentiAspectListRel);
        }
        logger.log(Level.DEBUG, "Calculating final Pi END");
    }

    private void calculatePsiMatrix() {
        logger.log(Level.DEBUG, "Calculating Psi START");
        for (int d = 0; d < ctDocSenti.size(); d++) {
            ArrayList<ProbEle> sentiList;
            if (psiMatrix.size() <= d) {
                sentiList = new ArrayList<>();
                psiMatrix.add(sentiList);
            } else {
                sentiList = psiMatrix.get(d);
            }

            int[] ctSenti = ctDocSenti.get(d);
            int ctSentiSum = ctDocSentiSum.get(d);

            for (int s = 0; s < ctSenti.length; s++) {
                double psi = (ctSenti[s] + beta[s]) / (ctSentiSum + beta[0] + beta[1]);
                ProbEle senti;
                if (sentiList.size() <= s) {
                    senti = new ProbEle(s, psi);
                    sentiList.add(senti);
                } else {
                    senti = sentiList.get(s);
                    double prob = senti.getProb();
                    senti.setProb(psi + prob);
                }
            }
        }
        logger.log(Level.DEBUG, "Calculating Psi END");
    }

    private void calculateFinalPsiMatrix() {
        logger.log(Level.DEBUG, "Calculating final Psi START");
        for (int d = 0; d < ctDocSenti.size(); d++) {
            ArrayList<ProbEle> sentiList = psiMatrix.get(d);
            int[] ctSenti = ctDocSenti.get(d);
            for (int s = 0; s < ctSenti.length; s++) {
                ProbEle senti = sentiList.get(s);
                double psi = senti.getProb() / updateNum;
                senti.setProb(psi);
            }
        }
        logger.log(Level.DEBUG, "Calculating final Psi END");
    }

    private void calculatePsiMatrixQuery() {
        logger.log(Level.DEBUG, "Calculating Psi Query START");
        psiMatrix = new ArrayList<>();
        for (int d = 0; d < ctDocSenti.size(); d++) {
            ArrayList<ProbEle> aspectList = new ArrayList<>();
            int[] ctSenti = ctDocSenti.get(d);
            int ctSentiSum = ctDocSentiSum.get(d);

            for (int s = 0; s < ctSenti.length; s++) {
                double psi = (ctSenti[s] + beta[s]) / (ctSentiSum + beta[0] + beta[1]);
                ProbEle aspect = new ProbEle(s, psi);
                aspectList.add(aspect);
            }
            psiMatrix.add(aspectList);
        }
        logger.log(Level.DEBUG, "Calculating Psi Query END");
    }

    private void outputTopwords(int iter) {
        try {
            logger.log(Level.DEBUG, "Outputing Top Words START");
            BufferedWriter writer = new BufferedWriter(
                    new OutputStreamWriter(new FileOutputStream(OUTPUTTOPWORDS, true)));
            writer.write("Results of Iteration #" + iter + "\n");
            writer.write("\nSentiment-Aspect Clusters:\n");
            for (int s = 0; s < phiMatrix.size(); s++) {
                ArrayList<ArrayList<ProbEle>> aspectSenti = phiMatrix.get(s);
                for (int a = 0; a < aspectSenti.size(); a++) {
                    ArrayList<ProbEle> termList = aspectSenti.get(a);
                    writer.write("S[" + s + "-A[" + a + "]\n");
                    for (int t = 0; t < TOPN_WORD; t++) {
                        ProbEle term = termList.get(t);
                        String sTerm = idxTermMap.get(term.getIdx());
                        writer.write(sTerm);
                        if (t != TOPN_WORD - 1) {
                            writer.write(",");
                        }
                    }
                    writer.write("\n");
                    writer.flush();
                }
            }

            writer.write("\nSentiment-Aspect Clusters (relative order):\n");

            for (int s = 0; s < phiMatrixRel.size(); s++) {
                ArrayList<ArrayList<ProbEle>> aspectSenti = phiMatrixRel.get(s);
                for (int a = 0; a < aspectSenti.size(); a++) {
                    ArrayList<ProbEle> termList = aspectSenti.get(a);
                    writer.write("S[" + s + "-A[" + a + "]\n");
                    for (int t = 0; t < TOPN_WORD; t++) {
                        ProbEle term = termList.get(t);
                        String sTerm = idxTermMap.get(term.getIdx());
                        writer.write(sTerm);
                        if (t != TOPN_WORD - 1) {
                            writer.write(",");
                        }
                    }
                    writer.write("\n");
                    writer.flush();
                }
            }
            //
            writer.write("\nUser-Sentiment-Aspect Clusters:\n");

            for (int u = 0; u < piMatrix.size(); u++) {
                ArrayList<ArrayList<ProbEle>> sentiAspectList = piMatrix.get(u);
                writer.write("U[" + idxUserMap.get(u).trim() + "]\n");
                for (int s = 0; s < SENTINUM; s++) {
                    ArrayList<ProbEle> senti = sentiAspectList.get(s);
                    writer.write("\tS[" + s + "]\n\t  ");
                    for (int a = 0; a < ASPECTNUM; a++) {
                        ProbEle sentiAspect = senti.get(a);
                        int aspect = sentiAspect.getAspect();
                        double prob = sentiAspect.getProb();
                        writer.write("A[" + aspect + "]:" + prob);
                        if (a != ASPECTNUM - 1) {
                            writer.write(",");
                        } else {
                            writer.write("\n");
                        }
                    }
                }
                writer.write("\n");
                writer.flush();
            }

            writer.write("\nUser-Sentiment-Aspect Clusters (relative order):\n");

            for (int u = 0; u < piMatrixRel.size(); u++) {
                ArrayList<ArrayList<ProbEle>> sentiAspectList = piMatrixRel.get(u);
                writer.write("U[" + idxUserMap.get(u).trim() + "]\n");
                for (int s = 0; s < SENTINUM; s++) {
                    ArrayList<ProbEle> senti = sentiAspectList.get(s);
                    writer.write("\tS[" + s + "]\n\t  ");
                    for (int a = 0; a < ASPECTNUM; a++) {
                        ProbEle sentiAspect = senti.get(a);
                        int aspect = sentiAspect.getAspect();
                        double prob = sentiAspect.getProb();
                        writer.write("A[" + aspect + "]:" + prob);
                        if (a != ASPECTNUM - 1) {
                            writer.write(",");
                        } else {
                            writer.write("\n");
                        }
                    }
                }
                writer.write("\n");
                writer.flush();
            }

            writer.flush();
            writer.close();

            logger.log(Level.DEBUG, "Outputting Top Words END");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void trainQuery(int maxIter, boolean export) {
        getResources();
        System.out.println("Initialize the parameters...");
        initialize();
        int iterNum = 1;
        int iterState = 1;

        System.out.println("Begin Gibbs sampling process...\n");
        while (true) {
            if (iterNum >= maxIter) {
                if (updateNum > 0) {

                    System.out.println("     -- write results to file...");
                    calculatePhiMatrix();
                    calculatePiMatrix();
                    calculatePsiMatrix();
                    updateNum++;
                    calculateFinalPhiMatrix();
                    calculateFinalPiMatrix();
                    calculateFinalPsiMatrix();
                    outputTopwords(iterNum);

                    if (export) {
                        System.out.println("     -- write models to file...");
                        ObjectOutputStream out;
                        try {
                            out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(
                                    OUTPUTMODEL + "fold_" + FOLD + "_" + ASPECTNUM + "_userprofile-0.dat")));
                            out.writeObject(this.ctUserSentiAspect);
                            out.close();
                            out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(
                                    OUTPUTMODEL + "fold_" + FOLD + "_" + ASPECTNUM + "_userprofile-1.dat")));
                            out.writeObject(this.ctUserSentiAspectSum);
                            out.close();
                            out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(
                                    OUTPUTMODEL + "fold_" + FOLD + "_" + ASPECTNUM + "_term-0.dat")));
                            out.writeObject(this.ctSentiAspectTerm);
                            out.close();
                            out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(
                                    OUTPUTMODEL + "fold_" + FOLD + "_" + ASPECTNUM + "_term-1.dat")));
                            out.writeObject(this.ctSentiAspectTermSum);
                            out.close();
                            out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(
                                    OUTPUTMODEL + "fold_" + FOLD + "_" + ASPECTNUM + "_sentiWordCount.dat")));
                            out.writeObject(this.sentiWordCountSetList);
                            out.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
                break;
            }
            if (iterNum > BURN_IN) {
                if (stage.equals("Burn-in")) {
                    iterState = 1;
                }
                stage = "Regular";
                if (iterState % THIN_INTERVAL == 0) {
                    stage = "Sampling";
                    calculatePhiMatrix();
                    calculatePiMatrix();
                    calculatePsiMatrix();
                    updateNum++;
                }
            }

            System.out.print("  ----- " + stage + " iteration [" + iterState + "] -----");
            Instant startTime = Instant.now();
            gibbsSampling();
            long elapsedTime = Duration.between(startTime, Instant.now()).toMillis();
            System.out.print(" in " + elapsedTime / 1000 + "s\n");
            logger.log(Level.INFO,
                    "  ----- " + stage + " iteration [" + iterState + "] ----- in " + elapsedTime / 1000 + "s\n");

            iterNum++;
            iterState++;
        }

        System.out.println("Begin query on testing data...");
        this.szUserDoc.clear();
        this.ctDocSenti.clear();
        this.ctDocSentiSum.clear();
        initializeSenti();
        int iterQuery = 0;
        System.out.println("Begin Gibbs sampling process on testing data...");
        while (true) {
            System.out.println("  -----Iteration: [" + iterQuery + "]-----");
            iterQuery++;
            gibbsSamplingSenti();
            if (iterQuery >= 100) {
                getOccrence();
                backup();
                ArrayList<ArrayList<String>> res = new ArrayList<>();
                HashMap<String, ArrayList<Double>> resPos = new HashMap<>();
                HashMap<String, ArrayList<Double>> resNeg = new HashMap<>();
                ArrayList<String> idxList = new ArrayList<>();
                HashMap<String, Integer> idxUserMap = new HashMap<>();
                for (int iterAverage = 0; iterAverage < 10; iterAverage++) {
                    gibbsSamplingSenti();
                    calculatePsiMatrixQuery();
                    idxList.clear();
                    ArrayList<String> predictList = computeSentiment(resPos, resNeg, idxList, idxUserMap);
                    res.add(predictList);
                    copyBack();
                }
                computeAvgSentiment(res);
                return;
            }
        }
    }

    private void computeAvgSentiment(ArrayList<ArrayList<String>> predictAllList) {
        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(LOGSENTI)));

            int nSize = predictAllList.get(0).size();
            double countT = 0;
            double count = 0;
            double countPos = 0d;
            double countNeg = 0d;
            double countPrepos = 0d;
            double countPreneg = 0d;
            double countTP = 0d;
            double countTN = 0d;
            double countFP = 0d;
            double countFN = 0d;

            for (int i = 0; i < nSize; i++) {
                int nPos = 0;
                int nNeg = 0;
                String trueLabel = "";
                double prob = 0d;
                for (ArrayList<String> predictList : predictAllList) {
                    String[] tokens = predictList.get(i).split("[|]");
                    String preLabel = tokens[0];
                    prob += Double.parseDouble(tokens[2]);
                    if (trueLabel.trim().length() == 0) {
                        trueLabel = tokens[1];
                        if (trueLabel.equals("pos")) {
                            countPos++;
                        } else if (trueLabel.equals("neg")) {
                            countNeg++;
                        }
                    } else {
                        if (!trueLabel.equalsIgnoreCase(tokens[1])) {
                            System.out.println("LABEL ERROR!");
                        }
                    }

                    if (preLabel.equals("pos")) {
                        nPos++;
                    } else {
                        nNeg++;
                    }
                }
                String avgLabel;
                if (nPos >= nNeg) {
                    avgLabel = "pos";
                } else {
                    avgLabel = "neg";
                }

                if (avgLabel.equals("pos")) {
                    countPrepos++;
                    if (avgLabel.equals(trueLabel)) {
                        countTP++;
                        countT++;
                    } else {
                        countFP++;
                    }
                } else {
                    countPreneg++;
                    if (avgLabel.equals(trueLabel)) {
                        countTN++;
                        countT++;
                    } else {
                        countFN++;
                    }
                }

                writer.write(avgLabel + "|" + trueLabel + "|" + prob / (double) predictAllList.size() + "\n");
                writer.flush();
                count++;
            }

            double posPre = countTP / countPrepos;
            double posRecall = countTP / countPos;
            double negPre = countTN / countPreneg;
            double negRecall = countTN / countNeg;

            System.out.println("*************************************************************");
            System.out.println("TP: " + countTP);
            System.out.println("FP: " + countFP);
            System.out.println("TN: " + countTN);
            System.out.println("FN: " + countFN);
            System.out.println("postive precision: " + posPre);
            System.out.println("postive recall: " + posRecall);
            System.out.println("postive f1 score: " + 2 * (posPre * posRecall) / (posPre + posRecall));
            System.out.println("negative precision: " + negPre);
            System.out.println("negative recall: " + negRecall);
            System.out.println("negative f1 score: " + 2 * (negPre * negRecall) / (negPre + negRecall));
            System.out.println("average precision: " + (posPre + negPre) / 2);
            System.out.println("average recall: " + (posRecall + negRecall) / 2);
            System.out.println("average f1 score: " + ((2 * (posPre * posRecall) / (posPre + posRecall))
                    + (2 * (negPre * negRecall) / (negPre + negRecall))) / 2);
            System.out.println("accuracy: " + countT / count);
            System.out.println("positive accuracy: " + countTP / (countTP + countFN));
            System.out.println("negative accuracy: " + countTN / (countTN + countFP));
            System.out.println(
                    "average accuracy: " + (countTP / (countTP + countFN) + countTN / (countTN + countFP)) / 2);
            System.out.println("*************************************************************");

            writer.write("TP: " + countTP + "\n");
            writer.write("FP: " + countFP + "\n");
            writer.write("TN: " + countTN + "\n");
            writer.write("FN: " + countFN + "\n");
            writer.write("postive precision: " + posPre + "\n");
            writer.write("postive recall: " + posRecall + "\n");
            writer.write("postive f1 score: " + 2 * (posPre * posRecall) / (posPre + posRecall) + "\n");
            writer.write("negative precision: " + negPre + "\n");
            writer.write("negative recall: " + negRecall + "\n");
            writer.write("negative f1 score: " + 2 * (negPre * negRecall) / (negPre + negRecall) + "\n");
            writer.write("average precision: " + (posPre + negPre) / 2 + "\n");
            writer.write("average recall: " + (posRecall + negRecall) / 2 + "\n");
            writer.write("average f1 score: " + ((2 * (posPre * posRecall) / (posPre + posRecall))
                    + (2 * (negPre * negRecall) / (negPre + negRecall))) / 2 + "\n");
            writer.write("accuracy: " + countT / count + "\n");
            writer.write("positive accuracy: " + countTP / (countTP + countFN) + "\n");
            writer.write("negative accuracy: " + countTN / (countTN + countFP) + "\n");
            writer.write(
                    "average accuracy: " + (countTP / (countTP + countFN) + countTN / (countTN + countFP)) / 2 + "\n");

            writer.flush();
            writer.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void gibbsSamplingSenti() {

        int docThreshold = 0;

        ArrayList<String> queryFilePathList = new ArrayList<>();
        File dir = new File(QUERYDATA);
        File[] files = dir.listFiles();
        assert files != null;
        for (File f : files) {
            if (f.isFile()) {
                if (f.getPath().contains(".DS_Store")) {
                    continue;
                }
                queryFilePathList.add(f.getPath());
            }
        }

        for (String filePath : queryFilePathList) {

            int docCount = 0;

            int idxLastSlash = filePath.lastIndexOf("\\");
            String username = filePath.substring(idxLastSlash + 1);
            int userIdx = userIdxMap.get(username);
            Sentence[][] szDoc = szUserDocQuery.get(userIdx);
            thresholdMapTesting.put(userIdx, docThreshold);

            for (int docIdx = 0; docIdx < szDoc.length; docIdx++) {
                if (szDoc[docIdx] == null) {
                    break;
                }
                int[] ctSenti = ctDocSenti.get(docThreshold + docIdx);
                Integer ctSentiSum = ctDocSentiSum.get(docThreshold + docIdx);

                for (int sentenceIdx = 0; sentenceIdx < szDoc[docIdx].length; sentenceIdx++) {

                    Sentence sentence = szDoc[docIdx][sentenceIdx];

                    int aspect = sentence.getAspect();
                    int sentiment = sentence.getSentiment();

                    ctSenti[sentiment]--;
                    ctSentiSum--;
                    ctDocSentiSum.set(docThreshold + docIdx, ctSentiSum);

                    int[][] ctSentiAspect = ctUserSentiAspect.get(userIdx);
                    ctSentiAspect[sentiment][aspect]--;
                    int[] ctDocAspectSum = ctUserSentiAspectSum.get(userIdx);
                    ctDocAspectSum[sentiment]--;

                    double[][] ctAspectTerm = ctSentiAspectTerm.get(sentiment);
                    double[] ctSentiAspectSum = ctSentiAspectTermSum.get(sentiment);
                    for (Word w : sentence.getWordList()) {
                        int curWordIdx = w.getTermIdx();
                        ctAspectTerm[aspect][curWordIdx]--;
                        ctSentiAspectSum[aspect]--;

                        if (semanticMap.containsKey(curWordIdx)) {
                            ArrayList<Integer> promotionList = w.getPromotionList();
                            if (promotionList != null) {
                                for (int similarIdx : promotionList) {
                                    double reduced = ctAspectTerm[aspect][similarIdx] - WORD_PROMOTION;
                                    ctAspectTerm[aspect][similarIdx] = Math.max(0, reduced);
                                    double reduced_all = ctSentiAspectSum[aspect] - WORD_PROMOTION;
                                    ctSentiAspectSum[aspect] = Math.max(0, reduced_all);
                                }
                            }
                        }

                    }
                    double[][] probSentiTopic = new double[SENTINUM][ASPECTNUM];
                    double probSum = 0d;

                    HashSet<Word> wordSet = sentence.getWordSet();

                    for (int s = 0; s < SENTINUM; s++) {
                        ctAspectTerm = ctSentiAspectTerm.get(s);
                        ctSentiAspectSum = ctSentiAspectTermSum.get(s);

                        for (int a = 0; a < ASPECTNUM; a++) {
                            double part1 = (ctSentiAspect[s][a] + gamma[a])
                                    / (ctDocAspectSum[s] + ASPECTNUM * gamma[a]);
                            double part2 = (ctSenti[s] + beta[s]) / (ctSentiSum + beta[POSITIVE] + beta[NEGATIVE]);

                            Iterator<Word> iterWord = wordSet.iterator();
                            double part3 = 1d;

                            int allCount = 0;
                            double aspectSum = ctSentiAspectSum[a];
                            double alphaSum = alphaAll[s];

                            while (iterWord.hasNext()) {
                                Word w = iterWord.next();
                                int curWordIdx = w.getTermIdx();

                                int wordCount = sentence.getWordCount(curWordIdx);
                                double alpha;
                                if (w.getSentiLex() < 0) {
                                    alpha = ALPHAGENERAL;
                                } else {
                                    if (w.getSentiLex() == s) {
                                        alpha = ALPHALEXICON;
                                    } else {
                                        alpha = ALPHANONLEXICON;
                                    }
                                }

                                for (int c = 0; c < wordCount; c++) {
                                    part3 *= (ctAspectTerm[a][curWordIdx] + alpha + c)
                                            / (aspectSum + alphaSum + allCount);
                                    allCount++;
                                }
                            }

                            double mrfWeight = 0;
                            double lambda = LAMBDA;

                            probSentiTopic[s][a] = part1 * part2 * part3 * Math.exp(lambda * mrfWeight);
                            probSum += probSentiTopic[s][a];
                        }

                    }

                    double threshold = 0d;
                    double r = generator.nextDouble() * probSum;
                    boolean isFound = false;

                    int newAspect = -1;
                    int newSentiment = -1;

                    for (int s = 0; s < SENTINUM; s++) {
                        for (int a = 0; a < ASPECTNUM; a++) {
                            threshold += probSentiTopic[s][a];
                            if (r <= threshold) {
                                newAspect = a;
                                newSentiment = s;
                                isFound = true;
                                break;
                            }
                        }
                        if (isFound) {
                            break;
                        }
                    }

                    sentence.setAspect(newAspect);
                    sentence.setSentiment(newSentiment);
                    szDoc[docIdx][sentenceIdx] = sentence;

                    ctSenti[newSentiment]++;
                    ctSentiSum++;
                    ctDocSentiSum.set(docThreshold + docIdx, ctSentiSum);

                    ctSentiAspect[newSentiment][newAspect]++;
                    ctDocAspectSum[newSentiment]++;

                    ctAspectTerm = ctSentiAspectTerm.get(newSentiment);
                    ctSentiAspectSum = ctSentiAspectTermSum.get(newSentiment);
                    for (Word w : sentence.getWordList()) {
                        int curWordIdx = w.getTermIdx();
                        ctAspectTerm[newAspect][curWordIdx]++;
                        ctSentiAspectSum[newAspect]++;
                        ArrayList<Integer> promotionList = new ArrayList<>();
                        ArrayList<SemanticPair> similarList = semanticMap.get(curWordIdx);
                        if (semanticMap.containsKey(curWordIdx)) {

                            for (int i = 0; i < similarList.size(); i++) {
                                SemanticPair pair = similarList.get(i);
                                int similarIdx = pair.getWordIdx();
                                double similarity = pair.getSimilarity();
                                if (newSentiment == 1 && pair.getPosSimilarity() >= pair.getNegSimilarity()) {
                                    continue;
                                }

                                if (newSentiment == 0 && pair.getPosSimilarity() < pair.getNegSimilarity()) {
                                    continue;
                                }

                                if (similarity > EPSILON) {
                                    ctAspectTerm[newAspect][similarIdx] += WORD_PROMOTION;
                                    ctSentiAspectSum[newAspect] += WORD_PROMOTION;
                                    promotionList.add(similarIdx);
                                }
                                if (i == 4) {
                                    break;
                                }
                            }
                            w.setPromotionList(promotionList);
                        }
                    }

                }
                docCount++;
            }
            docThreshold += docCount;
        }
    }

    private void getOccrence() {
        HashMap<String, Integer[]> occurenceMap = new HashMap<>();
        ArrayList<String> queryFilePathList = new ArrayList<>();
        File dir = new File(QUERYDATA);
        File[] files = dir.listFiles();
        assert files != null;
        for (File f : files) {
            if (f.isFile()) {
                if (f.getPath().contains(".DS_Store")) {
                    continue;
                }
                queryFilePathList.add(f.getPath());
            }
        }
        for (String filePath : queryFilePathList) {

            int idxLastSlash = filePath.lastIndexOf("\\");
            String username = filePath.substring(idxLastSlash + 1);
            int userIdx = userIdxMap.get(username);
            Sentence[][] szDoc = szUserDocQuery.get(userIdx);

            ArrayList<String> currentUserRate = szUserDocRate.get(userIdx);
            HashMap<Integer, String> idxMapUser = idxMap.get(userIdx);

            for (int docIdx = 0; docIdx < szDoc.length; docIdx++) {

                if (szDoc[docIdx] == null) {
                    break;
                }

                Integer[] aspectSentimentOccurance = new Integer[SENTINUM * ASPECTNUM];
                Arrays.fill(aspectSentimentOccurance, 0);

                for (int sentenceIdx = 0; sentenceIdx < szDoc[docIdx].length; sentenceIdx++) {
                    Sentence sentence = szDoc[docIdx][sentenceIdx];
                    int sentiment = sentence.getSentiment();
                    int aspect = sentence.getAspect();
                    aspectSentimentOccurance[sentiment * ASPECTNUM + aspect]++;
                }

                String trueSentiment = currentUserRate.get(docIdx);
                if (trueSentiment.length() > 0) {
                    String idx = idxMapUser.get(docIdx);
                    occurenceMap.put(idx, aspectSentimentOccurance);
                }
            }
        }
    }

    private ArrayList<String> computeSentiment(HashMap<String, ArrayList<Double>> posMap,
                                               HashMap<String, ArrayList<Double>> negMap, ArrayList<String> idxList, HashMap<String, Integer> idxUserMap) {

        try {
            ArrayList<String> predictLabels = new ArrayList<>();
            ArrayList<String> queryFilePathList = new ArrayList<>();
            File dir = new File(QUERYDATA);
            File[] files = dir.listFiles();
            assert files != null;
            for (File f : files) {
                if (f.isFile()) {
                    if (f.getPath().contains(".DS_Store")) {
                        continue;
                    }
                    queryFilePathList.add(f.getPath());
                }
            }

            double countT = 0;
            double count = 0;
            for (String filePath : queryFilePathList) {

                int idxLastSlash = filePath.lastIndexOf("\\");
                String username = filePath.substring(idxLastSlash + 1);
                int userIdx = userIdxMap.get(username);
                Sentence[][] szDoc = szUserDocQuery.get(userIdx);
                ArrayList<String> currentUserRate = szUserDocRate.get(userIdx);
                int docThreshold = thresholdMapTesting.get(userIdx);
                HashMap<Integer, String> idxMapUser = idxMap.get(userIdx);
                for (int docIdx = 0; docIdx < szDoc.length; docIdx++) {

                    if (szDoc[docIdx] == null) {
                        break;
                    }
                    if (szDoc[docIdx].length == 0) {
                        continue;
                    }

                    ArrayList<ProbEle> sentiAspectList = psiMatrix.get(docThreshold + docIdx);
                    double pPos = sentiAspectList.get(0).getProb();
                    double pNeg = sentiAspectList.get(1).getProb();

                    int nPos = 0;
                    int nNeg = 0;

                    for (int sentenceIdx = 0; sentenceIdx < szDoc[docIdx].length; sentenceIdx++) {
                        Sentence sentence = szDoc[docIdx][sentenceIdx];
                        int sentiment = sentence.getSentiment();
                        if (sentiment == 0) {
                            nPos++;
                        } else {
                            nNeg++;
                        }
                    }
                    String predictSentiment;
                    String trueSentiment = currentUserRate.get(docIdx);
                    if (trueSentiment.length() > 0) {

                        String idx = idxMapUser.get(docIdx);
                        idxUserMap.put(idx, userIdx);
                        idxList.add(idx);
                        if (posMap.containsKey(idx)) {
                            ArrayList<Double> posList = posMap.get(idx);
                            posList.add(pPos);
                        } else {
                            ArrayList<Double> posList = new ArrayList<>();
                            posList.add(pPos);
                            posMap.put(idx, posList);
                        }

                        if (negMap.containsKey(idx)) {
                            ArrayList<Double> negList = negMap.get(idx);
                            negList.add(pNeg);
                        } else {
                            ArrayList<Double> negList = new ArrayList<>();
                            negList.add(pNeg);
                            negMap.put(idx, negList);
                        }
                        if (nPos >= nNeg) {
                            predictSentiment = "pos";
                        } else {
                            predictSentiment = "neg";
                        }
                        predictLabels.add(predictSentiment + "|" + trueSentiment + "|"
                                + (double) nPos / ((double) nPos + (double) nNeg));
                        if (predictSentiment.equals(trueSentiment)) {
                            countT++;
                        }
                        count++;
                    }
                }
            }
            System.out.println(countT / count);
            return predictLabels;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void backup() {

        ctDocSentiOri = new ArrayList<>();
        ctDocSentiSumOri = new ArrayList<>();

        for (int d = 0; d < ctDocSenti.size(); d++) {
            int[] ctSenti = ctDocSenti.get(d);
            int sentiSum = ctDocSentiSum.get(d);
            int[] ctSentiOri = new int[SENTINUM];

            System.arraycopy(ctSenti, 0, ctSentiOri, 0, SENTINUM);
            ctDocSentiOri.add(ctSentiOri);
            ctDocSentiSumOri.add(sentiSum);
        }

        ctUserSentiAspectOri = new ArrayList<>();
        ctUserSentiAspectSumOri = new ArrayList<>();

        for (int u = 0; u < ctUserSentiAspect.size(); u++) {
            int[][] ctSentiAspect = ctUserSentiAspect.get(u);
            int[] ctDocAspectSum = ctUserSentiAspectSum.get(u);
            int[][] ctSentiAspectOri = new int[SENTINUM][ASPECTNUM];
            int[] ctDocAspectSumOri = new int[SENTINUM];
            for (int s = 0; s < SENTINUM; s++) {
                for (int a = 0; a < ASPECTNUM; a++) {
                    ctDocAspectSumOri[s] = ctDocAspectSum[s];
                    ctSentiAspectOri[s][a] = ctSentiAspect[s][a];
                }
            }
            ctUserSentiAspectOri.add(ctSentiAspectOri);
            ctUserSentiAspectSumOri.add(ctDocAspectSumOri);
        }

        ctSentiAspectTermOri = new ArrayList<>();
        ctSentiAspectTermSumOri = new ArrayList<>();

        for (int s = 0; s < SENTINUM; s++) {
            double[][] ctAspectTerm = ctSentiAspectTerm.get(s);
            double[] ctSentiAspectSum = ctSentiAspectTermSum.get(s);
            double[][] ctAspectTermOri = new double[ASPECTNUM][TERMNUM];
            double[] ctSentiAspectSumOri = new double[ASPECTNUM];
            for (int a = 0; a < ASPECTNUM; a++) {
                ctSentiAspectSumOri[a] = ctSentiAspectSum[a];
                if (TERMNUM >= 0) System.arraycopy(ctAspectTerm[a], 0, ctAspectTermOri[a], 0, TERMNUM);
            }
            ctSentiAspectTermOri.add(ctAspectTermOri);
            ctSentiAspectTermSumOri.add(ctSentiAspectSumOri);
        }

        szUserDocQueryOri = new ArrayList<>();

        for (Sentence[][] szDoc : szUserDocQuery) {
            Sentence[][] szDocOri = new Sentence[80000][];
            for (int docIdx = 0; docIdx < szDoc.length; docIdx++) {
                if (szDoc[docIdx] == null) {
                    break;
                }
                szDocOri[docIdx] = new Sentence[szDoc[docIdx].length];

                for (int senIdx = 0; senIdx < szDoc[docIdx].length; senIdx++) {

                    if (szDoc[docIdx][senIdx] == null) {
                        break;
                    }
                    Sentence sen = szDoc[docIdx][senIdx];
                    ArrayList<Word> wordList = sen.getWordList();

                    int sentiment = sen.getSentiment();
                    int aspect = sen.getAspect();

                    ArrayList<Word> wordListOri = new ArrayList<>();

                    for (Word w : wordList) {
                        Word wOri = new Word();
                        wOri.setAspect(w.getAspect());
                        wOri.setSentiment(w.getSentiment());
                        wOri.setSentiLex(w.getSentiLex());
                        wOri.setTermIdx(w.getTermIdx());
                        wordListOri.add(wOri);
                    }

                    Sentence senOri = new Sentence(sentiment, aspect, wordListOri);
                    szDocOri[docIdx][senIdx] = senOri;
                }
            }
            szUserDocQueryOri.add(szDocOri);
        }
    }

    private void copyBack() {

        ctDocSenti = new ArrayList<>();
        ctDocSentiSum = new ArrayList<>();

        for (int d = 0; d < ctDocSentiOri.size(); d++) {
            int[] ctSentiOri = ctDocSentiOri.get(d);
            int sentiSumOri = ctDocSentiSumOri.get(d);
            int[] ctSenti = new int[SENTINUM];

            System.arraycopy(ctSentiOri, 0, ctSenti, 0, SENTINUM);
            ctDocSenti.add(ctSenti);
            ctDocSentiSum.add(sentiSumOri);
        }

        ctUserSentiAspect = new ArrayList<>();
        ctUserSentiAspectSum = new ArrayList<>();

        for (int u = 0; u < ctUserSentiAspectOri.size(); u++) {
            int[][] ctSentiAspect = new int[SENTINUM][ASPECTNUM];
            int[] ctDocAspectSum = new int[SENTINUM];

            int[][] ctSentiAspectOri = ctUserSentiAspectOri.get(u);
            int[] ctDocAspectSumOri = ctUserSentiAspectSumOri.get(u);

            for (int s = 0; s < SENTINUM; s++) {
                for (int a = 0; a < ASPECTNUM; a++) {
                    ctDocAspectSum[s] = ctDocAspectSumOri[s];
                    ctSentiAspect[s][a] = ctSentiAspectOri[s][a];
                }
            }
            ctUserSentiAspect.add(ctSentiAspect);
            ctUserSentiAspectSum.add(ctDocAspectSum);
        }

        ctSentiAspectTerm = new ArrayList<>();
        ctSentiAspectTermSum = new ArrayList<>();

        for (int s = 0; s < SENTINUM; s++) {
            double[][] ctAspectTerm = new double[ASPECTNUM][TERMNUM];
            double[] ctSentiAspectSum = new double[ASPECTNUM];
            double[][] ctAspectTermOri = ctSentiAspectTermOri.get(s);
            double[] ctSentiAspectSumOri = ctSentiAspectTermSumOri.get(s);
            for (int a = 0; a < ASPECTNUM; a++) {
                ctSentiAspectSum[a] = ctSentiAspectSumOri[a];
                if (TERMNUM >= 0) System.arraycopy(ctAspectTermOri[a], 0, ctAspectTerm[a], 0, TERMNUM);
            }
            ctSentiAspectTerm.add(ctAspectTerm);
            ctSentiAspectTermSum.add(ctSentiAspectSum);
        }

        szUserDocQuery = new ArrayList<>();

        for (Sentence[][] szDocOri : szUserDocQueryOri) {
            Sentence[][] szDoc = new Sentence[80000][];
            for (int docIdx = 0; docIdx < szDocOri.length; docIdx++) {
                if (szDocOri[docIdx] == null) {
                    break;
                }
                szDoc[docIdx] = new Sentence[szDocOri[docIdx].length];

                for (int senIdx = 0; senIdx < szDocOri[docIdx].length; senIdx++) {

                    if (szDocOri[docIdx][senIdx] == null) {
                        break;
                    }
                    Sentence senOri = szDocOri[docIdx][senIdx];
                    ArrayList<Word> wordListOri = senOri.getWordList();

                    int sentimentOri = senOri.getSentiment();
                    int aspectOri = senOri.getAspect();

                    ArrayList<Word> wordList = new ArrayList<>();
                    for (Word wOri : wordListOri) {
                        Word w = new Word();
                        w.setAspect(wOri.getAspect());
                        w.setSentiment(wOri.getSentiment());
                        w.setSentiLex(wOri.getSentiLex());
                        w.setTermIdx(wOri.getTermIdx());
                        wordList.add(w);
                    }

                    Sentence sen = new Sentence(sentimentOri, aspectOri, wordList);
                    szDoc[docIdx][senIdx] = sen;
                }
            }
            szUserDocQuery.add(szDoc);
        }
    }

    private void initializeSenti() {
        try {
            thresholdMapTesting = new HashMap<>();
            szUserDocRate = new ArrayList<>();
            szUserDocQuery = new ArrayList<>();

            ctDocSenti = new ArrayList<>();
            ctDocSentiSum = new ArrayList<>();

            ArrayList<String> queryFilePathList = new ArrayList<>();
            File dir = new File(QUERYDATA);
            File[] files = dir.listFiles();
            assert files != null;
            for (File f : files) {
                if (f.isFile()) {
                    if (f.getPath().contains(".DS_Store")) {
                        continue;
                    }
                    queryFilePathList.add(f.getPath());
                }
            }

            idxMap = new HashMap<>();

            for (String filePath : queryFilePathList) {
                String line;
                BufferedReader reader = new BufferedReader(new FileReader(filePath));
                int idxLastSlash = filePath.lastIndexOf("\\");
                String username = filePath.substring(idxLastSlash + 1);
                int userIdx = userIdxMap.get(username);
                Sentence[][] szDoc = new Sentence[80000][];

                int[][] ctSentiAspect = ctUserSentiAspect.get(userIdx);
                int[] ctDocAspectSum = ctUserSentiAspectSum.get(userIdx);

                ArrayList<String> currentUserRate = new ArrayList<>();
                HashMap<Integer, String> idxMapUser = new HashMap<>();

                while ((line = reader.readLine()) != null) {
                    if (line.contains("[IDX]:")) {
                        int docIdx = -1;
                        int numSentence = -1;

                        Pattern pIdx = Pattern
                                .compile("\\[IDX]:(.*?)\\|\\[ID]:(.*?)\\|\\[SEN]:(.*?)\\|\\[R]:(.*?)\\|(.*?)");
                        Matcher mIdx = pIdx.matcher(line);
                        double rate = -1d;
                        String rId = "";
                        if (mIdx.matches()) {
                            docIdx = Integer.parseInt(mIdx.group(1));
                            rId = mIdx.group(2);
                            numSentence = Integer.parseInt(mIdx.group(3));
                            rate = Double.parseDouble(mIdx.group(4));
                        }
                        String sRate = "";

                        idxMapUser.put(docIdx, rId);

                        if (rate < 3) {
                            sRate = "neg";
                        } else if (rate >= 3) {
                            sRate = "pos";
                        } else {
                            System.out.println("negative rating!");
                        }

                        currentUserRate.add(sRate);

                        szDoc[docIdx] = new Sentence[numSentence];

                        int[] ctSenti = new int[SENTINUM];
                        int ctSentiSum = 0;

                        int sentIdx = 0;
                        while ((line = reader.readLine()).trim().length() != 0) {

                            String[] tokens = line.split("\\|");
                            line = tokens[0];

                            line = line.replaceAll("\\[[0-9]+]\\[[0-9]+]", "");

                            double rAspect = generator.nextDouble() * ASPECTNUM;
                            double rSentiment = generator.nextDouble() * SENTINUM;

                            int aspect = (int) rAspect;
                            int sentiment = (int) rSentiment;

                            ArrayList<Word> wordList = new ArrayList<>();
                            String[] words = line.split("[\\s]+");

                            for (String word : words) {
                                String curWord = idxTermMap.get(Integer.valueOf(word));
                                int curWordIdx = termIdxMap.get(curWord);
                                Word w = new Word();
                                w.setTermIdx(curWordIdx);
                                wordList.add(w);
                                if (!idxWordMap.containsKey(w.getTermIdx())) {
                                    idxWordMap.put(w.getTermIdx(), w);
                                }
                            }

                            ctSenti[sentiment]++;
                            ctSentiSum++;
                            ctSentiAspect[sentiment][aspect]++;
                            ctDocAspectSum[sentiment]++;
                            double[][] ctAspectTerm = ctSentiAspectTerm.get(sentiment);
                            double[] ctSentiAspectSum = ctSentiAspectTermSum.get(sentiment);

                            for (Word w : wordList) {
                                int curWordIdx = w.getTermIdx();
                                w.setAspect(aspect);
                                w.setSentiment(sentiment);
                                ctAspectTerm[aspect][curWordIdx]++;
                                ctSentiAspectSum[aspect]++;
                            }

                            Sentence sent = new Sentence(sentiment, aspect, wordList);
                            szDoc[docIdx][sentIdx] = sent;
                            sentIdx++;
                        }
                        ctDocSenti.add(ctSenti);
                        ctDocSentiSum.add(ctSentiSum);
                    }

                }
                szUserDocRate.add(currentUserRate);
                reader.close();
                szUserDocQuery.add(szDoc);
                idxMap.put(userIdx, idxMapUser);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        Options options = new Options();
        options.addOption("a", true, "Number of aspects");
        options.addOption("b1", true, "Beta of positive");
        options.addOption("b2", true, "Beta of negative");
        options.addOption("g", true, "gamma");
        options.addOption("a1", true, "alpha of general");
        options.addOption("a2", true, "Number of lexicon");
        options.addOption("a3", true, "Number of non lexicon");
        options.addOption("l", true, "Discourse promotion");
        options.addOption("r", true, "Word promotion");
        options.addOption("i", true, "Number of iterations");
        options.addOption("n", true, "Number of burn-in iterations");
        options.addOption("q", true, "Number of query iterations");
        options.addOption("t", true, "Thin interval");
        options.addOption("o", false, "Output model files");
        CommandLineParser clp = new DefaultParser();
        CommandLine cl = null;

        int aspectNum = ASPECTNUM;
        double beta1 = BETA_POSITIVE;
        double beta2 = BETA_NEGATIVE;
        double gammaValue = GAMMA;
        double alphaGeneral = ALPHAGENERAL;
        double alphaLexicon = ALPHALEXICON;
        double alphaNonlexicon = ALPHANONLEXICON;
        double lambda = LAMBDA;
        double rho = WORD_PROMOTION;
        double epsilon = EPSILON;
        int burnIn = BURN_IN;
        int thinInterval = THIN_INTERVAL;
        int maxIter = MAX_ITERATION;
        int maxIterTesting = MAX_TEST_ITERATION;
        boolean outputModel = GENERATE_MODELS;

        try {
            cl = clp.parse(options, args);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
        }
        if (args.length == 0) {
            (new HelpFormatter()).printHelp(80, "Limbic [options]", "", options, "");
            System.err.println("No args specified, use default values");
        }

        assert cl != null;
        if (cl.hasOption("a")) {
            aspectNum = Integer.parseInt(cl.getOptionValue("a"));
        }

        if (cl.hasOption("b1")) {
            beta1 = Double.parseDouble(cl.getOptionValue("b1"));
        }

        if (cl.hasOption("b2")) {
            beta2 = Double.parseDouble(cl.getOptionValue("b2"));
        }

        if (cl.hasOption("g")) {
            gammaValue = Double.parseDouble(cl.getOptionValue("g"));
        }

        if (cl.hasOption("a1")) {
            alphaGeneral = Double.parseDouble(cl.getOptionValue("a1"));
        }

        if (cl.hasOption("a2")) {
            alphaLexicon = Double.parseDouble(cl.getOptionValue("a2"));
        }

        if (cl.hasOption("a3")) {
            alphaNonlexicon = Double.parseDouble(cl.getOptionValue("a3"));
        }

        if (cl.hasOption("l")) {
            lambda = Double.parseDouble(cl.getOptionValue("l"));
        }

        if (cl.hasOption("r")) {
            rho = Double.parseDouble(cl.getOptionValue("r"));
        }

        if (cl.hasOption("e")) {
            epsilon = Double.parseDouble(cl.getOptionValue("e"));
        }

        if (cl.hasOption("i")) {
            maxIter = Integer.parseInt(cl.getOptionValue("i"));
        }

        if (cl.hasOption("n")) {
            burnIn = Integer.parseInt(cl.getOptionValue("n"));
        }

        if (cl.hasOption("q")) {
            maxIterTesting = Integer.parseInt(cl.getOptionValue("q"));
        }

        if (cl.hasOption("t")) {
            thinInterval = Integer.parseInt(cl.getOptionValue("t"));
        }

        if (cl.hasOption("o")) {
            outputModel = true;
        }

        TraitModel trait = new TraitModel(aspectNum, beta1, beta2, gammaValue, alphaGeneral, alphaLexicon,
                alphaNonlexicon, lambda, rho, epsilon, maxIter, maxIterTesting, burnIn, thinInterval, outputModel);
        Instant startTime = Instant.now();
        trait.trainQuery(MAX_ITERATION, GENERATE_MODELS);
        long elapsedTime = Duration.between(startTime, Instant.now()).toMillis() / 1000;
        System.out.println("Finished in " + elapsedTime + "s\n");
        logger.log(Level.INFO, "Finished in " + elapsedTime + "s\n");
    }
}
