package edu.ucsc.cs.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.evaluation.statistics.filter.AtomFilter;
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.ObservedAtom;
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.util.database.Queries;

public class Evaluator {
	
	private final Database result;
	private Database baseline;
	private Predicate p;

	
	public Evaluator(Database result, Database baseline, Predicate p) {
		this.result = result;
		this.baseline = baseline;
		this.p = p;
		
	}
	
	public void outputToFile(){
		BufferedWriter writer = null;
		String dir = "/Users/dhanyasridhar/Documents/psl-stance/";
		String groundTruthFile = p.toString() + "_truth.csv";
		String resultsFile = p.toString() + "_inference_results.csv";
		try {
			writer = new BufferedWriter(new FileWriter(resultsFile));
			
			for (GroundAtom atom : Queries.getAllAtoms(result, p)){
				GroundTerm[] terms = atom.getArguments();
				writer.append(terms[0] + "," + terms[1] + "," + atom.getValue() + "\n");
                                writer.flush();
			}
			
			writer = new BufferedWriter(new FileWriter(groundTruthFile));
			
			for (GroundAtom atom : Queries.getAllAtoms(baseline, p)){
				GroundTerm[] terms = atom.getArguments();
				writer.append(terms[0] + "," + terms[1] + "," + atom.getValue() + "\n");
                                writer.flush();
			}
			writer.close();

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
