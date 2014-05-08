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


//dataSet = "fourforums"
dataSet = "stance-classification"
ConfigManager cm = ConfigManager.getManager()
ConfigBundle cb = cm.getBundle(dataSet)

def defaultPath = System.getProperty("java.io.tmpdir")
//String dbPath = cb.getString("dbPath", defaultPath + File.separator + "psl-" + dataSet)
String dbPath = cb.getString("dbPath", defaultPath + File.separator + dataSet)
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbPath, true), cb)

PSLModel model = new PSLModel(this, data)

/*
 * Author predicates of the form: predicate(authorID, authorID, topic) 
 * or (authorID, topic) 
 * or (authorID, postID)
 * Observed predicates
 */

model.add predicate: "writesPost" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "participates" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "agreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "disagreesAuth" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Author predicates for social attitudes e.g. sarcasm, nasty, attack
 */
model.add predicate: "sarcastic" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "nasty" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "attacks" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Post level observed predicates
 */

model.add predicate: "hasTopic" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "hasLabelPro" , types:[ArgumentType.UniqueID, ArgumentType.String]

/*
 * Auxiliary topic predicate
 */
model.add predicate: "topic" , types:[ArgumentType.String]


/*
 * Latent, open predicates for latent network
 */

model.add predicate: "supports" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "against" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]

/*
 * Target predicates
 */
model.add predicate: "isProAuth" , types:[ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "isProPost" , types:[ArgumentType.UniqueID, ArgumentType.String]


//model.add predicate: "agreesPost" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]
//model.add predicate: "disagreesPost" , types:[ArgumentType.UniqueID, ArgumentType.UniqueID]


/*
 * Rule expressing that an author and their post will have the same stances and same agreement behavior 
 * Note that the second is logically equivalent to saying that if author is pro then post will be pro - contrapositive
 */

model.add rule : (isProPost(P, T) & writesPost(A, P)) >> isProAuth(A, T), weight : 1
model.add rule : (~(isProPost(P, T)) & writesPost(A, P) & hasTopic(P, T))>> ~(isProAuth(A, T)), weight : 1

/*
 * Rules for relating stance with agreement/disagreement
 */

/*
model.add rule : (agreesPost(P1, P2) & (P1^P2) & isProPost(P1, T)) >> isProPost(P2, T), weight : 1
model.add rule : (agreesPost(P1, P2) & (P1^P2) & ~(isProPost(P1, T))) >> ~(isProPost(P2, T)), weight : 1
model.add rule : (~(agreesPost(P1, P2)) & (P1^P2) & isProPost(P1, T)) >> ~(isProPost(P2, T)), weight : 1
model.add rule : (~(agreesPost(P1, P2)) & (P1^P2) & ~(isProPost(P1, T))) >> isProPost(P2, T), weight : 1
*/
/*
model.add rule : (agreesAuth(A1, A2, T) & (A1^A2) & isProAuth(A1, T)) >> isProAuth(A2, T), weight : 1
model.add rule : (agreesAuth(A1, A2, T) & (A1^A2) & ~(isProAuth(A1, T))) >> ~(isProAuth(A2, T)), weight : 1
model.add rule : (disagreesAuth(A1, A2, T) & (A1^A2) & isProAuth(A1, T)) >> ~(isProAuth(A2, T)), weight : 1
model.add rule : (disagreesAuth(A1, A2, T) & (A1^A2) & topic(T) & ~(isProAuth(A1, T))) >> isProAuth(A2, T), weight : 1
*/

/*
 * Propagating stance with the inferred network
 * Add participates predicate as a clause
 * second rule is actually propagating stance from B -> A
 * encode an actual isAnti predicate
 * small development dataset
 */

model.add rule : (supports(A1, A2, T) & (A1 - A2) & isProAuth(A1, T)) >> isProAuth(A2, T), weight : 1
model.add rule : (supports(A1, A2, T) & (A1 - A2) & ~(isProAuth(A1, T))) >> ~(isProAuth(A2, T)), weight : 1
model.add rule : (against(A1, A2, T) & (A1 - A2) & isProAuth(A1, T)) >> ~(isProAuth(A2, T)), weight : 1
model.add rule : (against(A1, A2, T) & (A1 - A2) & topic(T) & ~(isProAuth(A1, T))) >> isProAuth(A2, T), weight : 1

model.add rule : (supports(A1, A2, T) & (A1 - A2) & isProAuth(A2, T)) >> isProAuth(A1, T), weight : 1
model.add rule : (supports(A1, A2, T) & (A1 - A2) & ~(isProAuth(A2, T))) >> ~(isProAuth(A1, T)), weight : 1
model.add rule : (against(A1, A2, T) & (A1 - A2) & isProAuth(A2, T)) >> ~(isProAuth(A1, T)), weight : 1
model.add rule : (against(A1, A2, T) & (A1 - A2) & topic(T) & ~(isProAuth(A2, T))) >> isProAuth(A1, T), weight : 1

/*
 * Cross-topic agreement and disagreement
 */
model.add rule : (agreesAuth(A1, A2, T) & (A1 - A2) & participates(A1, T2) & participates(A2, T2)) >> supports(A1, A2, T2), weight : 1
model.add rule : (disagreesAuth(A1, A2, T) & (A1 - A2) & participates(A1, T2) & participates(A2, T2)) >> against(A1, A2, T2), weight : 1

model.add rule : (agreesAuth(A1, A2, T) & (A1 - A2)) >> supports(A1, A2, T) , weight : 1
model.add rule : (disagreesAuth(A1, A2, T) & (A1 - A2)) >> against(A1, A2, T) , weight : 1

/*
 * Rules relating sarcasm to against
 */
model.add rule : (sarcastic(A1, A2, P, T) & (A1 - A2)) >> against(A1, A2, T) , weight : 1
model.add rule : (sarcastic(A1, A2, P, T) & (A1 - A2) & participates(A1, T2) & participates(A2, T2)) >> against(A1, A2, T2), weight : 1

/*
 * Rules relating nastiness to against
 */
model.add rule : (nasty(A1, A2, T) & (A1 - A2)) >> against(A1, A2, T) , weight : 1
model.add rule : (nasty(A1, A2, T) & (A1 - A2) & participates(A1, T2) & participates(A2, T2)) >> against(A1, A2, T2), weight : 1

/*
 * Rules relating attacks to against
 */
model.add rule : (attacks(A1, A2, T) & (A1 - A2)) >> against(A1, A2, T) , weight : 1
model.add rule : (attacks(A1, A2, T) & (A1 - A2) & participates(A1, T2) & participates(A2, T2)) >> against(A1, A2, T2), weight : 1


/*
 * Rules for propagating disagreement/agreement through the network
 * Doesn't do anything since agreement predicates are closed
 
model.add rule : (agreesAuth(P1, P2) & (P1^P2) & agreesAuth(P2, P3) & (P2^P3) & (P1^P3)) >> agreesAuth(P1, P3), weight : 5
model.add rule : (~(agreesAuth(P1, P2)) & (P1^P2) & agreesAuth(P2, P3) & (P2^P3) & (P1^P3)) >> ~(agreesAuth(P1, P3)), weight : 5
model.add rule : (agreesAuth(P1, P2) & (P1^P2) & ~(agreesAuth(P2, P3)) & (P2^P3) & (P1^P3)) >> ~(agreesAuth(P1, P3)), weight : 5
model.add rule : (~(agreesAuth(P1, P2)) & (P1^P2) & ~(agreesAuth(P2, P3)) & (P2^P3) & (P1^P3)) >> agreesAuth(P1, P3), weight : 5
*/

/*
 * Transitivity/triad rules for supports/against
 */

/*
model.add rule : (supports(A1, A2, T) & supports(A2, A3, T) & (A1 ^ A2) & (A2 ^ A3)) >> supports(A1, A3, T) , weight : 1
model.add rule : (supports(A1, A2, T) & against(A2, A3, T) & (A1 ^ A2) & (A2 ^ A3)) >> against(A1, A3, T) , weight : 1

model.add rule : (against(A1, A2, T) & against(A2, A3, T) & (A1 ^ A2) & (A2 ^ A3)) >> supports(A1, A3, T) , weight : 1
model.add rule : (against(A1, A2, T) & supports(A2, A3, T) & (A1 ^ A2) & (A2 ^ A3)) >> against(A1, A3, T) , weight : 1

model.add rule : (supports(A1, A2, T) & supports(A3, A2, T) & (A1 ^ A2) & (A1 ^ A3)) >> supports(A1, A3, T), weight : 1
model.add rule : (supports(A1, A2, T) & against(A3, A2, T) & (A1 ^ A2) & (A1 ^ A3)) >> against(A1, A3, T), weight : 1

model.add rule : (against(A1, A2, T) & supports(A3, A2, T) & (A1 ^ A2) & (A1 ^ A3)) >> against(A1, A3, T), weight : 1
model.add rule : (against(A1, A2, T) & against(A3, A2, T) & (A1 ^ A2) & (A1 ^ A3)) >> supports(A1, A3, T), weight : 1
*/
//Prior that the label given by the text classifier is indeed the stance label

model.add rule : (hasLabelPro(P, T)) >> isProPost(P, T) , weight : 1
model.add rule : (~(hasLabelPro(P, T))) >> ~(isProPost(P, T)) , weight : 1

/*
 * Inserting data into the data store
 */
fold = 1

foldStr = "fold" + String.valueOf(fold) + java.io.File.separator;

Partition observed_tr = new Partition(0);
Partition predict_tr = new Partition(1);
Partition truth_tr = new Partition(2);
Partition observed_te = new Partition(3);
Partition predict_te = new Partition(4);
Partition truth_te = new Partition(5);
Partition dummy_tr = new Partition(6);
Partition dummy_tr2 = new Partition(7);
Partition dummy_te = new Partition(8);
Partition dummy_te2 = new Partition(9);

def dir = 'data'+java.io.File.separator+ foldStr + 'train'+java.io.File.separator;

inserter = data.getInserter(hasLabelPro, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"labels.csv", ",");

inserter = data.getInserter(hasTopic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"post_topics.csv", ",");

inserter = data.getInserter(writesPost, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"author_posts.csv", ",");

inserter = data.getInserter(topic, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"topics.csv", ",");

inserter = data.getInserter(participates, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"authortopic.csv", ",")

inserter = data.getInserter(agreesAuth, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"authoragreement.csv",",");

inserter = data.getInserter(disagreesAuth, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"authordisagreement.csv", ",");

inserter = data.getInserter(sarcastic, observed_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"sarcasm.csv", ",");

inserter = data.getInserter(nasty, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"nastiness.csv", ",");

inserter = data.getInserter(attacks, observed_tr)
InserterUtils.loadDelimitedData(inserter, dir+"attack.csv", ",");


/*
 * Ground truth for training data for weight learning
 */

inserter = data.getInserter(isProPost, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"post_pro.csv",",");

inserter = data.getInserter(isProAuth, truth_tr)
InserterUtils.loadDelimitedDataTruth(inserter, dir+"authorpro.csv", ",");

/*
 * Used later on to populate training DB with all possible interactions
 */

inserter = data.getInserter(supports, dummy_tr)
InserterUtils.loadDelimitedData(inserter, dir + "interaction.csv", ",")

inserter = data.getInserter(against, dummy_tr2)
InserterUtils.loadDelimitedData(inserter, dir + "interaction.csv", ",")

/*
 * Testing split for model inference
 * Observed partitions
 */

def testdir = 'data'+java.io.File.separator+ foldStr + 'test'+java.io.File.separator;

inserter = data.getInserter(hasLabelPro, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"labels.csv", ",");

inserter = data.getInserter(hasTopic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"post_topics.csv", ",");

inserter = data.getInserter(writesPost, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"author_posts.csv",",");

inserter = data.getInserter(topic, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"topics.csv",",");

inserter = data.getInserter(participates, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"authortopic.csv",",")

inserter = data.getInserter(agreesAuth, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"authoragreement.csv",",");

inserter = data.getInserter(disagreesAuth, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"authordisagreement.csv", ",");

inserter = data.getInserter(sarcastic, observed_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"sarcasm.csv", ",");

inserter = data.getInserter(nasty, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"nastiness.csv", ",");

inserter = data.getInserter(attacks, observed_te)
InserterUtils.loadDelimitedData(inserter, testdir+"attack.csv", ",");

/*
 * Random variable partitions
 */

inserter = data.getInserter(isProPost, truth_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"post_pro.csv",",");

inserter = data.getInserter(isProAuth, truth_te)
InserterUtils.loadDelimitedDataTruth(inserter, testdir+"authorpro.csv", ",");

inserter = data.getInserter(supports, dummy_te)
InserterUtils.loadDelimitedData(inserter, testdir + "interaction.csv", ",")

inserter = data.getInserter(against, dummy_te2)
InserterUtils.loadDelimitedData(inserter, testdir + "interaction.csv", ",")


/*
 * Set up training databases for weight learning using training set
 */

Database distributionDB = data.getDatabase(predict_tr, [sarcastic, nasty, attacks, agreesAuth, disagreesAuth, participates, hasLabelPro, hasTopic, writesPost, topic] as Set, observed_tr);
Database truthDB = data.getDatabase(truth_tr, [isProPost, isProAuth] as Set)
Database dummy_DB = data.getDatabase(dummy_tr, [supports] as Set)
Database dummy_DB2 = data.getDatabase(dummy_tr2, [against] as Set)

/* Populate isProPost in observed DB. */
DatabasePopulator dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(truthDB, isProPost);


/* Populate isProAuth in observed DB. */
DatabasePopulator populator = new DatabasePopulator(distributionDB);
populator.populateFromDB(truthDB, isProAuth);

/*
 * Populate distribution DB with all possible interactions
 */
dbPop = new DatabasePopulator(distributionDB);
dbPop.populateFromDB(dummy_DB, supports);

dbPop.populateFromDB(dummy_DB2, against);

//MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(model, distributionDB, truthDB, cb);
HardEM weightLearning = new HardEM(model, distributionDB, truthDB, cb);
println "about to start weight learning"
weightLearning.learn();
println " finished weight learning "
weightLearning.close();

/*
 MaxPseudoLikelihood mple = new MaxPseudoLikelihood(model, trainDB, truthDB, cb);
 println "about to start weight learning"
 mple.learn();
 println " finished weight learning "
 mlpe.close();
 */

println model;

Database testDB = data.getDatabase(predict_te, [sarcastic, nasty, attacks, agreesAuth, disagreesAuth, participates, hasLabelPro, hasTopic, writesPost, topic] as Set, observed_te);
Database testTruthDB = data.getDatabase(truth_te, [isProPost, isProAuth] as Set)

Database dummy_test = data.getDatabase(dummy_te, [supports] as Set)
Database dummy_test2 = data.getDatabase(dummy_te2, [against] as Set)

/* Populate isProPost in test DB. */

DatabasePopulator test_pop = new DatabasePopulator(testDB);
test_pop.populateFromDB(testTruthDB, isProPost);


/* Populate isProAuth in test DB. */

DatabasePopulator test_populator = new DatabasePopulator(testDB);
test_populator.populateFromDB(testTruthDB, isProAuth);

test_populator.populateFromDB(dummy_test, supports);
test_populator.populateFromDB(dummy_test2, against);

/*
 * Inference
 */

MPEInference mpe = new MPEInference(model, testDB, cb)
FullInferenceResult result = mpe.mpeInference()
System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

/* Evaluation */

def comparator = new DiscretePredictionComparator(testDB)
comparator.setBaseline(testTruthDB)
comparator.setResultFilter(new MaxValueFilter(isProPost, 1))
comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero

Set<GroundAtom> groundings = Queries.getAllAtoms(testTruthDB, isProPost)
int totalTestExamples = groundings.size()
DiscretePredictionStatistics stats = comparator.compare(isProPost, totalTestExamples)
System.out.println("Accuracy: " + stats.getAccuracy())
System.out.println("F1: " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Precision: " + stats.getPrecision(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Recall: " + stats.getRecall(DiscretePredictionStatistics.BinaryClass.POSITIVE))

comparator.setResultFilter(new MaxValueFilter(isProAuth, 1))
comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero

Set<GroundAtom> authorGroundings = Queries.getAllAtoms(testTruthDB, isProAuth)
totalTestExamples = authorGroundings.size()
DiscretePredictionStatistics authorstats = comparator.compare(isProAuth, totalTestExamples)
System.out.println("Accuracy: " + authorstats.getAccuracy())
System.out.println("F1: " + authorstats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Precision: " + authorstats.getPrecision(DiscretePredictionStatistics.BinaryClass.POSITIVE))
System.out.println("Recall: " + authorstats.getRecall(DiscretePredictionStatistics.BinaryClass.POSITIVE))

testTruthDB.close()
testDB.close()
distributionDB.close()
truthDB.close()
