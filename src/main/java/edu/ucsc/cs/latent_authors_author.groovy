package edu.ucsc.cs

import java.util.Set;
import edu.umd.cs.bachuai13.util.DataOutputter;
import edu.umd.cs.bachuai13.util.FoldUtils;
import edu.umd.cs.bachuai13.util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin.NormScalingType
import edu.umd.cs.psl.application.learning.weight.random.FirstOrderMetropolisRandOM
import edu.umd.cs.psl.application.learning.weight.random.HardEMRandOM
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.application.learning.weight.em.DualEM

import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter

import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator

import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.Model
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.Queries
import edu.ucsc.cs.utils.Evaluator;

import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.database.ReadOnlyDatabase;


//dataSet = "fourforums"
dataSet = "stance-classification"
ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
//String dbPath = cb.getString("dbPath", defaultPath + File.separator + "psl-" + dataSet)
String dbPath = cb.getString("dbPath", defaultPath + File.separator + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbPath, true), cb)

fold = args[1]
def dir = 'data'+java.io.File.separator + fold + java.io.File.separator + 'train' + java.io.File.separator;
def testdir = 'data'+java.io.File.separator + fold + java.io.File.separator + 'test' + java.io.File.separator;

initialWeight = 5

PSLModel model = new PSLModel(this, data)

/*
 * Author predicates of the form: predicate(authorID, authorID, topic) 
 * or (authorID, topic) 
 * or (authorID, postID)
 * Observed predicates
 */

model.add predicate: "writesPost" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "participates" , types:[ArgumentType.UniqueID, ArgumentType.String]
//model.add predicate: "agreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
//model.add predicate: "disagreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Author predicates for social attitudes e.g. sarcasm, nasty, attack
 */
/*
model.add predicate: "sarcastic" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "nasty" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "attacks" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "agrees" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
*/
model.add predicate: "sarcastic" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "nasty" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "attacks" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "agrees" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]

/*
 * Post level observed predicates
 */

model.add predicate: "hasTopic" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "hasLabelPro" , types:[ArgumentType.UniqueID, ArgumentType.String]

/*
 * Auxiliary topic predicate
 */
model.add predicate: "topic" , types:[ArgumentType.String]
model.add predicate: "ideology" , types:[ArgumentType.String]

/* Latent ideology predicate */
model.add predicate: "hasIdeologyA" , types:[ArgumentType.UniqueID]

/*
 * Latent, open predicates for latent network
 */

model.add predicate: "supports" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "valInt" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String] 
//model.add predicate: "against" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Target predicates
 */
model.add predicate: "isProAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "isProPost" , types:[ArgumentType.UniqueID, ArgumentType.String]


/*
 * Rule expressing that an author and their post will have the same stances and same agreement behavior 
 * Note that the second is logically equivalent to saying that if author is pro then post will be pro - contrapositive
 */

model.add rule : (isProPost(P, T) & writesPost(A, P)) >> isProAuth(A, T), weight : initialWeight
model.add rule : (isProAuth(A, T) & writesPost(A, P) & hasTopic(P, T)) >> isProPost(P, T), weight :initialWeight
model.add rule : (~isProPost(P, T) & writesPost(A, P) & hasTopic(P,T)) >> ~isProAuth(A, T), weight : initialWeight
model.add rule : (~isProAuth(A, T) & writesPost(A, P) & hasTopic(P, T)) >> ~isProPost(P, T), weight : initialWeight

//model.add rule : (isProPost(P, T) & writesPost(A, P)) >> isProAuth(A, T), constraint:true
//model.add rule : (isProAuth(A, T) & writesPost(A, P) & hasTopic(P, T)) >> isProPost(P, T), constraint:true
//model.add rule : (~isProPost(P, T) & writesPost(A, P) & hasTopic(P,T)) >> ~isProAuth(A, T), constraint:true
//model.add rule : (~isProAuth(A, T) & writesPost(A, P) & hasTopic(P, T)) >> ~isProPost(P, T), constraint:true


/* simple stance rules*/


model.add rule : (agrees(P1, P2, A1, A2) & (A1-A2) & hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
model.add rule : (agrees(P1, P2, A1, A2) & (A1-A2) & ~hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight

model.add rule : (sarcastic(P1, P2, A1, A2) & (A1-A2) & ~hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
model.add rule : (sarcastic(P1, P2, A1, A2) & (A1-A2) & hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight

model.add rule : (nasty(P1, P2, A1, A2) & (A1-A2) & ~hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
model.add rule : (nasty(P1, P2, A1, A2) & (A1-A2) & hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight

model.add rule : (attacks(P1, P2, A1, A2) & (A1-A2) & ~hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
model.add rule : (attacks(P1, P2, A1, A2) & (A1-A2) & hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight

/*
 * Propagating stance with the inferred network
 * Add participates predicate as a clause
 * second rule is actually propagating stance from B -> A
 * encode an actual isAnti predicate
 * small development dataset
 */

/*
 * Rules relating attitudes to stance
 */

/*
model.add rule : (agrees(P1, P2, A1, A2) & (P1 - P2) & (A1 - A2) & participates(A1, T) & participates(A2, T)) >> supports(A1, A2, T), weight : initialWeight
model.add rule : (sarcastic(P1, P2, A1, A2) & (P1 - P2) & (A1 - A2) & participates(A1, T) & participates(A2, T)) >> ~supports(A1, A2, T) , weight : initialWeight
model.add rule : (nasty(P1, P2, A1, A2) & (P1 - P2) & (A1 - A2) & participates(A1, T) & participates(A2, T)) >> ~supports(A1, A2, T) , weight : initialWeight
model.add rule : (attacks(P1, P2, A1, A2) & (P1 - P2) & (A1 - A2) & participates(A1, T) & participates(A2, T)) >> ~supports(A1, A2, T) , weight : initialWeight

model.add rule : (supports(A1, A2, T) & (A1 - A2) & participates(A1, T2) & participates(A2, T2)) >> supports(A1, A2, T2), weight : initialWeight
model.add rule : (~supports(A1, A2, T) & valInt(A1, A2, T) & valInt(A1, A2, T2) & (A1 - A2) & participates(A1, T) & participates(A2, T) & participates(A1, T2) & participates(A2, T2)) >> ~supports(A1, A2, T2), weight : initialWeight

model.add rule : (supports(A1, A2, T) & (A1 - A2) & participates(A2, T) & isProAuth(A1, T)) >> isProAuth(A2, T), weight : initialWeight
model.add rule : (supports(A1, A2, T) & (A1 - A2) & participates(A1, T) & topic(T) & ~(isProAuth(A1, T))) >> ~(isProAuth(A2, T)), weight : initialWeight
model.add rule : (~supports(A1, A2, T) & (A1 - A2) & valInt(A1, A2, T) & participates(A2, T) & participates(A1, T) & isProAuth(A1, T)) >> ~isProAuth(A2, T), weight : initialWeight
model.add rule : (~supports(A1, A2, T) & (A1 - A2) & valInt(A1, A2, T) & participates(A2, T) & participates(A1, T) & ~isProAuth(A1, T)) >> isProAuth(A2, T), weight : initialWeight
*/

/*
model.add rule : (supports(A1, A2, T) & (A1 - A2) & hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
model.add rule : (~supports(A1, A2, T) & valInt(A1, A2, T) & (A1 - A2) & hasIdeologyA(A1)) >> hasIdeologyB(A2), weight : initialWeight

model.add rule : (supports(A1, A2, T) & (A1 - A2) & hasIdeologyB(A1)) >> hasIdeologyB(A2), weight : initialWeight
model.add rule : (~supports(A1, A2, T) & valInt(A1, A2, T) & (A1 - A2) & hasIdeologyB(A1)) >> hasIdeologyA(A2), weight : initialWeight
*/


model.add rule : (hasIdeologyA(A1) & hasIdeologyA(A2) & (A1 - A2) & isProAuth(A1, T) & participates(A2, T)) >> isProAuth(A2, T), weight : initialWeight
model.add rule : (hasIdeologyA(A1) & hasIdeologyA(A2) & (A1 - A2) & ~isProAuth(A1, T) & participates(A1, T) & participates(A2, T)) >> ~isProAuth(A2, T), weight : initialWeight

model.add rule : (isProAuth(A1, T) & isProAuth(A2, T)  & (A1 - A2) & hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight
model.add rule : (~isProAuth(A1, T) & ~isProAuth(A2, T) & (A1 - A2) & participates(A1, T) & participates(A2, T) & hasIdeologyA(A1)) >> hasIdeologyA(A2), weight : initialWeight

//model.add rule : (~hasIdeologyA(A1) & ~hasIdeologyA(A2) & (A1 - A2) & isProAuth(A1, T) & participates(A2, T)) >> isProAuth(A2, T), weight : initialWeight
//model.add rule : (~hasIdeologyA(A1) & ~hasIdeologyA(A2) & (A1 - A2) & ~isProAuth(A1, T) & participates(A1, T) & participates(A2, T)) >> ~isProAuth(A2, T), weight : initialWeight

//model.add rule : (isProAuth(A1, T) & isProAuth(A2, T)  & (A1 - A2) & ~hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight
//model.add rule : (~isProAuth(A1, T) & ~isProAuth(A2, T) & (A1 - A2) & participates(A1, T) & participates(A2, T) & ~hasIdeologyA(A1)) >> ~hasIdeologyA(A2), weight : initialWeight

//Prior that the label given by the text classifier is indeed the stance label

model.add rule : (hasLabelPro(P, T)) >> isProPost(P, T) , weight : initialWeight
model.add rule : (~(hasLabelPro(P, T))) >> ~(isProPost(P, T)) , weight : initialWeight

/*
 * Inserting data into the data store
 */

/* training partitions */
Partition observed_tr = new Partition(0);
Partition predict_tr = new Partition(1);
Partition truth_tr = new Partition(2);
Partition dummy_tr = new Partition(3);

/*testing partitions */
Partition observed_te = new Partition(4);
Partition predict_te = new Partition(5);
Partition dummy_te = new Partition(6);

/*separate partitions for the gold standard truth for testing */
Partition postProTruth = new Partition(7);
Partition postAntiTruth = new Partition(8);
Partition authProTruth = new Partition(9);
Partition authAntiTruth = new Partition(10);

inserter = data.getInserter(hasLabelPro, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"hasLabelPro.csv", ",");

//inserter = data.getInserter(isProPost, observed_tr)
//InserterUtils.loadDelimitedDataTruth(inserter, dir+"hasLabelPro.csv", ",");


inserter = data.getInserter(hasTopic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"hasTopic.csv", ",");

inserter = data.getInserter(writesPost, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"writesPost.csv", ",");

inserter = data.getInserter(topic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"topic.csv", ",");

inserter = data.getInserter(participates, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"participates.csv", ",")

/*load sentiment predicates with soft truth values*/

inserter = data.getInserter(agrees, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"agreement_binary_verbose.csv",",");

inserter = data.getInserter(sarcastic, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"sarcasm_binary_verbose.csv", ",");

inserter = data.getInserter(nasty, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"nastiness_binary_verbose.csv", ",");

inserter = data.getInserter(attacks, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"attack_binary_verbose.csv", ",");

inserter = data.getInserter(valInt, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"supports.csv", ",");

inserter = data.getInserter(hasIdeologyA, observed_tr)
inserter.insertValue(1.0, "a624")
inserter.insertValue(0.0, "a859")


/*
 * Ground truth for training data for weight learning
 */

inserter = data.getInserter(isProPost, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"isProPost.csv",",");

inserter = data.getInserter(isProAuth, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"isProAuth.csv", ",");

/*
 * Used later on to populate training DB with all possible interactions
 */

inserter = data.getInserter(supports, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "supports.csv", ",")

inserter = data.getInserter(hasIdeologyA, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "hasIdeologyA.csv", ",")


/*db population for all possible stance atoms*/

inserter = data.getInserter(isProAuth, dummy_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir + "isProAuth.csv", ",")

inserter = data.getInserter(isProPost, dummy_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir + "isProPost.csv", ",")

/*
 * Testing split for model inference
 * Observed partitions
 */

inserter = data.getInserter(hasLabelPro, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"hasLabelPro.csv", ",");

//inserter = data.getInserter(isProPost, observed_te)
//InserterUtils.loadDelimitedDataTruth(inserter, testdir+"hasLabelPro.csv", ",");

inserter = data.getInserter(hasTopic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"hasTopic.csv", ",");

inserter = data.getInserter(writesPost, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"writesPost.csv",",");

inserter = data.getInserter(topic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"topic.csv",",");

inserter = data.getInserter(participates, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"participates.csv",",");

inserter = data.getInserter(agrees, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"agreement_binary_verbose.csv",",");

inserter = data.getInserter(sarcastic, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"sarcasm_binary_verbose.csv", ",");

inserter = data.getInserter(nasty, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"nastiness_binary_verbose.csv", ",");

inserter = data.getInserter(attacks, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"attack_binary_verbose.csv", ",");
/*
inserter = data.getInserter(agreesAuth, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"agreesAuth.csv",",");

inserter = data.getInserter(disagreesAuth, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"disagreesAuth.csv", ",");

inserter = data.getInserter(sarcastic, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"sarcasm.csv", ",");

inserter = data.getInserter(nasty, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"nastiness.csv", ",");

inserter = data.getInserter(attacks, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"attack.csv", ",");
*/

inserter = data.getInserter(valInt, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"supports.csv", ",");

inserter = data.getInserter(hasIdeologyA, observed_te)
inserter.insertValue(1.0, "a799");
inserter.insertValue(1.0, "a118");

/*
 * Random variable partitions
 */

inserter = data.getInserter(isProPost, postProTruth)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"isProPost.csv",",");

inserter = data.getInserter(isProAuth, authProTruth)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"isProAuth.csv", ",");

/*supports and against*/

inserter = data.getInserter(supports, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "supports.csv", ",")

inserter = data.getInserter(hasIdeologyA, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "hasIdeologyA.csv", ",")


/*to populate testDB with the correct rvs */
inserter = data.getInserter(isProAuth, dummy_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "isProAuth.csv", ",")

inserter = data.getInserter(isProPost, dummy_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir + "isProPost.csv", ",")


/*
 * Set up training databases for weight learning using training set
 */

Database distributionDB = data.getDatabase(predict_tr, [hasLabelPro, sarcastic, nasty, attacks, agrees, participates, hasTopic, writesPost, topic] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [isProPost, isProAuth] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [hasIdeologyA, supports, isProAuth, isProPost] as Set)

/* Populate distribution DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, isProPost);
dbPop.populateFromDB(dummy_DB, isProAuth);

/*
 * Populate distribution DB with all possible interactions
 */
dbPop.populateFromDB(dummy_DB, supports);
dbPop.populateFromDB(dummy_DB, hasIdeologyA);


DualEM weightLearning = new DualEM(model, distributionDB, truthDB, cb);
weightLearning.learn();
weightLearning.close();

println model;

Database testDB = data.getDatabase(predict_te, [hasLabelPro, sarcastic, nasty, attacks, agrees, participates, hasTopic, writesPost, topic] as Set, observed_te);

Database testTruth_postPro = data.getDatabase(postProTruth, [isProPost] as Set)
Database testTruth_authPro = data.getDatabase(authProTruth, [isProAuth] as Set)

Database dummy_test = data.getDatabase(dummy_te, [hasIdeologyA, supports, isProAuth, isProPost] as Set)

/* Populate in test DB. */

DatabasePopulator test_populator = new DatabasePopulator(testDB);
test_populator.populateFromDB(dummy_test, isProAuth);
test_populator.populateFromDB(dummy_test, isProPost);

test_populator.populateFromDB(dummy_test, supports);
test_populator.populateFromDB(dummy_test, hasIdeologyA);

/*
 * Inference
 */

MPEInference mpe = new MPEInference(model, testDB, cb)
FullInferenceResult result = mpe.mpeInference();

/*output prediction results */
Evaluator evaluator = new Evaluator(testDB, isProPost, "psl_ideology", fold);
evaluator.outputToFile();

///*output prediction results */
//evaluator = new Evaluator(testDB, supports, "supports", fold);
//evaluator.outputToFile();

/*output prediction results */
evaluator = new Evaluator(testDB, hasIdeologyA, "ideologyA_authors_author", fold);
evaluator.outputToFile();

/* Accuracy */
def discComp = new DiscretePredictionComparator(testDB)
discComp.setBaseline(testTruth_postPro)
discComp.setResultFilter(new MaxValueFilter(isProPost, 1))
discComp.setThreshold(0.5) // treat best value as true as long as it is nonzero

Set<GroundAtom> groundings = Queries.getAllAtoms(testTruth_postPro, isProPost)
int totalTestExamples = groundings.size()
DiscretePredictionStatistics stats = discComp.compare(isProPost, totalTestExamples)
System.out.println("Accuracy: " + stats.getAccuracy())

def comparator = new SimpleRankingComparator(testDB)
comparator.setBaseline(testTruth_postPro)

// Choosing what metrics to report
def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,  RankingScore.AreaROC]
double [] score = new double[metrics.size()]

try {
    for (int i = 0; i < metrics.size(); i++) {
            comparator.setRankingScore(metrics.get(i))
            score[i] = comparator.compare(isProPost)
    }
    //Storing the performance values of the current fold
    System.out.println(fold + "," + score[0] + "," + score[1] + "," + score[2])

//    System.out.println("\nArea under positive-class PR curve: " + score[0])
//    System.out.println("Area under negetive-class PR curve: " + score[1])
//    System.out.println("Area under ROC curve: " + score[2])
}
catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("No evaluation data! Terminating!");
}

comparator = new SimpleRankingComparator(testDB)
comparator.setBaseline(testTruth_authPro)

// Choosing what metrics to report
metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,  RankingScore.AreaROC]
score = new double[metrics.size()]

try {
    for (int i = 0; i < metrics.size(); i++) {
            comparator.setRankingScore(metrics.get(i))
            score[i] = comparator.compare(isProAuth)
    }
    //Storing the performance values of the current fold
    System.out.println(fold + "," + score[0] + "," + score[1] + "," + score[2])

//    System.out.println("\nArea under positive-class PR curve: " + score[0])
//    System.out.println("Area under negetive-class PR curve: " + score[1])
//    System.out.println("Area under ROC curve: " + score[2])
}
catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("No evaluation data! Terminating!");
}

testDB.close()
distributionDB.close()
truthDB.close()
dummy_test.close()
dummy_DB.close()
testTruth_postPro.close()
testTruth_authPro.close()