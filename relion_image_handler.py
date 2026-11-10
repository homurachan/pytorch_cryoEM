#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# /***************************************************************************
#  *
#  * Author: "Sjors H.W. Scheres"
#  * MRC Laboratory of Molecular Biology
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * This complete copyright notice must be included in any revised version of the
#  * source code. Additional authorship citations may be added, but existing
#  * author citations must be preserved.
#  ***************************************************************************/
# Python adaptation, 2026. GPL-2.0-or-later. Based on the supplied RELION 3.1.2 SIRM v1p2.
# Standalone: no sibling modules, RELION executable, or WSL required.
# /***************************************************************************
#  *
#  * Authors:    Roberto Marabini	(roberto@cnb.csic.es)
#  *             Carlos Oscar S. Sorzano	(coss@cnb.csic.es)
#  *
#  * Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307 USA
#  *
#  * All comments concerning this program package may be sent to the
#  * e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
# /***************************************************************************
#  *
#  * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#  *
#  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  *  All comments concerning this program package may be sent to the
#  *  e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
# /***************************************************************************
#  *
#  * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
#  *              Sjors H.W. Scheres
#  *
#  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  *  All comments concerning this program package may be sent to the
#  *  e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
# /***************************************************************************
#  *
#  * Authors:		 J.R. Bilbao-Castro (jrbcast@ace.ual.es)
#  *
#  * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
#  *
#  * This program is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program; if not, write to the Free Software
#  * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#  * 02111-1307  USA
#  *
#  *	All comments concerning this program package may be sent to the
#  *	e-mail address 'xmipp@cnb.csic.es'
#  ***************************************************************************/
LABEL_TYPES = {'randomSeed': 'INT', 'rlnComment': 'STRING', 'rlnAreaId': 'INT', 'rlnAreaName': 'STRING', 'rlnBodyMaskName': 'STRING', 'rlnBodyKeepFixed': 'INT', 'rlnBodyReferenceName': 'STRING', 'rlnBodyRotateDirectionX': 'DOUBLE', 'rlnBodyRotateDirectionY': 'DOUBLE', 'rlnBodyRotateDirectionZ': 'DOUBLE', 'rlnBodyRotateRelativeTo': 'INT', 'rlnBodySigmaAngles': 'DOUBLE', 'rlnBodySigmaOffset': 'DOUBLE', 'rlnBodySigmaOffsetAngst': 'DOUBLE', 'rlnBodySigmaRot': 'DOUBLE', 'rlnBodySigmaTilt': 'DOUBLE', 'rlnBodySigmaPsi': 'DOUBLE', 'rlnBodyStarFile': 'STRING', 'rlnCtfAstigmatism': 'DOUBLE', 'rlnCtfBfactor': 'DOUBLE', 'rlnCtfMaxResolution': 'DOUBLE', 'rlnCtfValidationScore': 'DOUBLE', 'rlnCtfScalefactor': 'DOUBLE', 'rlnVoltage': 'DOUBLE', 'rlnDefocusU': 'DOUBLE', 'rlnDefocusV': 'DOUBLE', 'rlnDefocusAngle': 'DOUBLE', 'rlnSphericalAberration': 'DOUBLE', 'rlnChromaticAberration': 'DOUBLE', 'rlnDetectorPixelSize': 'DOUBLE', 'rlnCtfPowerSpectrum': 'STRING', 'rlnEnergyLoss': 'DOUBLE', 'rlnCtfFigureOfMerit': 'DOUBLE', 'rlnCtfImage': 'STRING', 'rlnLensStability': 'DOUBLE', 'rlnMagnification': 'DOUBLE', 'rlnPhaseShift': 'DOUBLE', 'rlnConvergenceCone': 'DOUBLE', 'rlnLongitudinalDisplacement': 'DOUBLE', 'rlnTransversalDisplacement': 'DOUBLE', 'rlnAmplitudeContrast': 'DOUBLE', 'rlnCtfValue': 'DOUBLE', 'rlnImageName': 'STRING', 'rlnImageOriginalName': 'STRING', 'rlnReconstructImageName': 'STRING', 'rlnImageId': 'INT', 'rlnEnabled': 'BOOL', 'rlnDataType': 'INT', 'rlnImageDimensionality': 'INT', 'rlnBeamTiltX': 'DOUBLE', 'rlnBeamTiltY': 'DOUBLE', 'rlnMtfFileName': 'STRING', 'rlnOpticsGroup': 'INT', 'rlnOpticsGroupName': 'STRING', 'rlnOddZernike': 'DOUBLE_VECTOR', 'rlnEvenZernike': 'DOUBLE_VECTOR', 'rlnImagePixelSize': 'DOUBLE', 'rlnMagMat00': 'DOUBLE', 'rlnMagMat01': 'DOUBLE', 'rlnMagMat10': 'DOUBLE', 'rlnMagMat11': 'DOUBLE', 'rlnCoordinateX': 'DOUBLE', 'rlnCoordinateY': 'DOUBLE', 'rlnCoordinateZ': 'DOUBLE', 'rlnMovieFrameNumber': 'INT', 'rlnNormCorrection': 'DOUBLE', 'rlnMagnificationCorrection': 'DOUBLE', 'rlnSamplingRate': 'DOUBLE', 'rlnSamplingRateX': 'DOUBLE', 'rlnSamplingRateY': 'DOUBLE', 'rlnSamplingRateZ': 'DOUBLE', 'rlnImageSize': 'INT', 'rlnImageSizeX': 'INT', 'rlnImageSizeY': 'INT', 'rlnImageSizeZ': 'INT', 'rlnMinimumValue': 'DOUBLE', 'rlnMaximumValue': 'DOUBLE', 'rlnAverageValue': 'DOUBLE', 'rlnStandardDeviationValue': 'DOUBLE', 'rlnSkewnessValue': 'DOUBLE', 'rlnKurtosisExcessValue': 'DOUBLE', 'rlnImageWeight': 'DOUBLE', 'rlnMaskName': 'STRING', 'rlnJobIsContinue': 'BOOL', 'rlnJobType': 'INT', 'rlnJobTypeName': 'STRING', 'rlnJoboptionType': 'INT', 'rlnJobOptionVariable': 'STRING', 'rlnJobOptionValue': 'STRING', 'rlnJobOptionGUILabel': 'STRING', 'rlnJobOptionDefaultValue': 'STRING', 'rlnJobOptionSliderMin': 'DOUBLE', 'rlnJobOptionSliderMax': 'DOUBLE', 'rlnJobOptionSliderStep': 'DOUBLE', 'rlnJobOptionHelpText': 'STRING', 'rlnJobOptionFilePattern': 'STRING', 'rlnJobOptionDirectoryDefault': 'STRING', 'rlnJobOptionMenuOptions': 'STRING', 'rlnMatrix_1_1': 'DOUBLE', 'rlnMatrix_1_2': 'DOUBLE', 'rlnMatrix_1_3': 'DOUBLE', 'rlnMatrix_2_1': 'DOUBLE', 'rlnMatrix_2_2': 'DOUBLE', 'rlnMatrix_2_3': 'DOUBLE', 'rlnMatrix_3_1': 'DOUBLE', 'rlnMatrix_3_2': 'DOUBLE', 'rlnMatrix_3_3': 'DOUBLE', 'rlnAccumMotionTotal': 'DOUBLE', 'rlnAccumMotionEarly': 'DOUBLE', 'rlnAccumMotionLate': 'DOUBLE', 'rlnMicrographId': 'INT', 'rlnMicrographName': 'STRING', 'rlnMicrographGainName': 'STRING', 'rlnMicrographDefectFile': 'STRING', 'rlnMicrographNameNoDW': 'STRING', 'rlnMicrographMovieName': 'STRING', 'rlnMicrographMetadata': 'STRING', 'rlnMicrographTiltAngle': 'DOUBLE', 'rlnMicrographTiltAxisDirection': 'DOUBLE', 'rlnMicrographTiltAxisOutOfPlane': 'DOUBLE', 'rlnMicrographOriginalPixelSize': 'DOUBLE', 'rlnMicrographPixelSize': 'DOUBLE', 'rlnMicrographPreExposure': 'DOUBLE', 'rlnMicrographDoseRate': 'DOUBLE', 'rlnMicrographBinning': 'DOUBLE', 'rlnMicrographFrameNumber': 'INT', 'rlnMotionModelVersion': 'INT', 'rlnMicrographStartFrame': 'INT', 'rlnMicrographEndFrame': 'INT', 'rlnMicrographShiftX': 'DOUBLE', 'rlnMicrographShiftY': 'DOUBLE', 'rlnMotionModelCoeffsIdx': 'INT', 'rlnMotionModelCoeff': 'DOUBLE', 'rlnEERUpsampling': 'INT', 'rlnEERGrouping': 'INT', 'rlnAccuracyRotations': 'DOUBLE', 'rlnAccuracyTranslations': 'DOUBLE', 'rlnAccuracyTranslationsAngst': 'DOUBLE', 'rlnAveragePmax': 'DOUBLE', 'rlnCurrentResolution': 'DOUBLE', 'rlnCurrentImageSize': 'INT', 'rlnSsnrMap': 'DOUBLE', 'rlnReferenceDimensionality': 'INT', 'rlnDataDimensionality': 'INT', 'rlnDiff2RandomHalves': 'DOUBLE', 'rlnEstimatedResolution': 'DOUBLE', 'rlnFourierCompleteness': 'DOUBLE', 'rlnOverallFourierCompleteness': 'DOUBLE', 'rlnGoldStandardFsc': 'DOUBLE', 'rlnGroupName': 'STRING', 'rlnGroupNumber': 'INT', 'rlnGroupNrParticles': 'INT', 'rlnGroupScaleCorrection': 'DOUBLE', 'rlnNrHelicalAsymUnits': 'INT', 'rlnHelicalTwist': 'DOUBLE', 'rlnHelicalTwistMin': 'DOUBLE', 'rlnHelicalTwistMax': 'DOUBLE', 'rlnHelicalTwistInitialStep': 'DOUBLE', 'rlnHelicalRise': 'DOUBLE', 'rlnHelicalRiseMin': 'DOUBLE', 'rlnHelicalRiseMax': 'DOUBLE', 'rlnHelicalRiseInitialStep': 'DOUBLE', 'rlnIsHelix': 'BOOL', 'rlnFourierSpaceInterpolator': 'INT', 'rlnLogLikelihood': 'DOUBLE', 'rlnMinRadiusNnInterpolation': 'INT', 'rlnNormCorrectionAverage': 'DOUBLE', 'rlnNrClasses': 'INT', 'rlnNrBodies': 'INT', 'rlnNrGroups': 'INT', 'rlnSpectralOrientabilityContribution': 'DOUBLE', 'rlnOriginalImageSize': 'INT', 'rlnPaddingFactor': 'DOUBLE', 'rlnClassDistribution': 'DOUBLE', 'rlnClassPriorOffsetX': 'DOUBLE', 'rlnClassPriorOffsetY': 'DOUBLE', 'rlnOrientationDistribution': 'DOUBLE', 'rlnPixelSize': 'DOUBLE', 'rlnReferenceSpectralPower': 'DOUBLE', 'rlnOrientationalPriorMode': 'INT', 'rlnReferenceImage': 'STRING', 'rlnSGDGradientImage': 'STRING', 'rlnSigmaOffsets': 'DOUBLE', 'rlnSigmaOffsetsAngst': 'DOUBLE', 'rlnSigma2Noise': 'DOUBLE', 'rlnReferenceSigma2': 'DOUBLE', 'rlnSigmaPriorRotAngle': 'DOUBLE', 'rlnSigmaPriorTiltAngle': 'DOUBLE', 'rlnSigmaPriorPsiAngle': 'DOUBLE', 'rlnSignalToNoiseRatio': 'DOUBLE', 'rlnTau2FudgeFactor': 'DOUBLE', 'rlnReferenceTau2': 'DOUBLE', 'rlnOverallAccuracyRotations': 'DOUBLE', 'rlnOverallAccuracyTranslations': 'DOUBLE', 'rlnOverallAccuracyTranslationsAngst': 'DOUBLE', 'rlnAdaptiveOversampleFraction': 'DOUBLE', 'rlnAdaptiveOversampleOrder': 'INT', 'rlnAutoLocalSearchesHealpixOrder': 'INT', 'rlnAvailableMemory': 'DOUBLE', 'rlnBestResolutionThusFar': 'DOUBLE', 'rlnCoarseImageSize': 'INT', 'rlnChangesOptimalOffsets': 'DOUBLE', 'rlnChangesOptimalOrientations': 'DOUBLE', 'rlnChangesOptimalClasses': 'DOUBLE', 'rlnCtfDataArePhaseFlipped': 'BOOL', 'rlnCtfDataAreCtfPremultiplied': 'BOOL', 'rlnExperimentalDataStarFile': 'STRING', 'rlnDoCorrectCtf': 'BOOL', 'rlnDoCorrectMagnification': 'BOOL', 'rlnDoCorrectNorm': 'BOOL', 'rlnDoCorrectScale': 'BOOL', 'rlnDoExternalReconstruct': 'BOOL', 'rlnDoRealignMovies': 'BOOL', 'rlnDoMapEstimation': 'BOOL', 'rlnDoStochasticGradientDescent': 'BOOL', 'rlnDoStochasticEM': 'BOOL', 'rlnExtReconsDataReal': 'STRING', 'rlnExtReconsDataImag': 'STRING', 'rlnExtReconsWeight': 'STRING', 'rlnExtReconsResult': 'STRING', 'rlnExtReconsResultStarfile': 'STRING', 'rlnDoFastSubsetOptimisation': 'BOOL', 'rlnSgdInitialIterations': 'INT', 'rlnSgdFinalIterations': 'INT', 'rlnSgdInBetweenIterations': 'INT', 'rlnSgdInitialResolution': 'DOUBLE', 'rlnSgdFinalResolution': 'DOUBLE', 'rlnSgdInitialSubsetSize': 'INT', 'rlnSgdFinalSubsetSize': 'INT', 'rlnSgdMuFactor': 'DOUBLE', 'rlnSgdSigma2FudgeInitial': 'DOUBLE', 'rlnSgdSigma2FudgeHalflife': 'INT', 'rlnSgdSkipAnneal': 'BOOL', 'rlnSgdSubsetSize': 'INT', 'rlnSgdWriteEverySubset': 'INT', 'rlnSgdMaxSubsets': 'INT', 'rlnSgdStepsize': 'DOUBLE', 'rlnDoAutoRefine': 'BOOL', 'rlnDoOnlyFlipCtfPhases': 'BOOL', 'rlnDoSolventFlattening': 'BOOL', 'rlnDoSolventFscCorrection': 'BOOL', 'rlnDoSkipAlign': 'BOOL', 'rlnDoSkipRotate': 'BOOL', 'rlnDoSplitRandomHalves': 'BOOL', 'rlnDoZeroMask': 'BOOL', 'rlnFixSigmaNoiseEstimates': 'BOOL', 'rlnFixSigmaOffsetEstimates': 'BOOL', 'rlnFixTauEstimates': 'BOOL', 'rlnHasConverged': 'BOOL', 'rlnHasHighFscAtResolLimit': 'BOOL', 'rlnHasLargeSizeIncreaseIterationsAgo': 'INT', 'rlnDoHelicalRefine': 'BOOL', 'rlnIgnoreHelicalSymmetry': 'BOOL', 'rlnFourierMask': 'STRING', 'rlnHelicalTwistInitial': 'DOUBLE', 'rlnHelicalRiseInitial': 'DOUBLE', 'rlnHelicalCentralProportion': 'DOUBLE', 'rlnNrHelicalNStart': 'INT', 'rlnHelicalMaskTubeInnerDiameter': 'DOUBLE', 'rlnHelicalMaskTubeOuterDiameter': 'DOUBLE', 'rlnHelicalSymmetryLocalRefinement': 'BOOL', 'rlnHelicalSigmaDistance': 'DOUBLE', 'rlnHelicalKeepTiltPriorFixed': 'BOOL', 'rlnLowresLimitExpectation': 'DOUBLE', 'rlnHighresLimitExpectation': 'DOUBLE', 'rlnHighresLimitSGD': 'DOUBLE', 'rlnDoIgnoreCtfUntilFirstPeak': 'BOOL', 'rlnIncrementImageSize': 'INT', 'rlnCurrentIteration': 'INT', 'rlnLocalSymmetryFile': 'STRING', 'rlnJoinHalvesUntilThisResolution': 'DOUBLE', 'rlnMagnificationSearchRange': 'DOUBLE', 'rlnMagnificationSearchStep': 'DOUBLE', 'rlnMaximumCoarseImageSize': 'INT', 'rlnMaxNumberOfPooledParticles': 'INT', 'rlnModelStarFile': 'STRING', 'rlnModelStarFile2': 'STRING', 'rlnNumberOfIterations': 'INT', 'rlnNumberOfIterWithoutResolutionGain': 'INT', 'rlnNumberOfIterWithoutChangingAssignments': 'INT', 'rlnOpticsStarFile': 'STRING', 'rlnOutputRootName': 'STRING', 'rlnParticleDiameter': 'DOUBLE', 'rlnRadiusMaskMap': 'INT', 'rlnRadiusMaskExpImages': 'INT', 'rlnRandomSeed': 'INT', 'rlnRefsAreCtfCorrected': 'BOOL', 'rlnSmallestChangesClasses': 'INT', 'rlnSmallestChangesOffsets': 'DOUBLE', 'rlnSmallestChangesOrientations': 'DOUBLE', 'rlnOrientSamplingStarFile': 'STRING', 'rlnSolventMaskName': 'STRING', 'rlnSolventMask2Name': 'STRING', 'rlnTauSpectrumName': 'STRING', 'rlnUseTooCoarseSampling': 'BOOL', 'rlnWidthMaskEdge': 'INT', 'rlnIsFlip': 'BOOL', 'rlnOrientationsID': 'INT', 'rlnOriginX': 'DOUBLE', 'rlnOriginY': 'DOUBLE', 'rlnOriginZ': 'DOUBLE', 'rlnOriginXPrior': 'DOUBLE', 'rlnOriginYPrior': 'DOUBLE', 'rlnOriginZPrior': 'DOUBLE', 'rlnOriginXAngst': 'DOUBLE', 'rlnOriginYAngst': 'DOUBLE', 'rlnOriginZAngst': 'DOUBLE', 'rlnOriginXPriorAngst': 'DOUBLE', 'rlnOriginYPriorAngst': 'DOUBLE', 'rlnOriginZPriorAngst': 'DOUBLE', 'rlnAngleRot': 'DOUBLE', 'rlnAngleRotPrior': 'DOUBLE', 'rlnAngleRotFlipRatio': 'DOUBLE', 'rlnAngleTilt': 'DOUBLE', 'rlnAngleTiltPrior': 'DOUBLE', 'rlnAnglePsi': 'DOUBLE', 'rlnAnglePsiPrior': 'DOUBLE', 'rlnAnglePsiFlipRatio': 'DOUBLE', 'rlnAnglePsiFlip': 'BOOL', 'rlnAngleRotNear': 'DOUBLE', 'rlnAngleTiltNear': 'DOUBLE', 'rlnAngleDeltaTheta': 'DOUBLE', 'rlnWeightVectorX': 'DOUBLE', 'rlnWeightVectorY': 'DOUBLE', 'rlnWeightVectorZ': 'DOUBLE', 'rlnWeightVectorValue': 'DOUBLE', 'rlnDeltaZ': 'DOUBLE', 'rlnParticleSerialNumber': 'INT', 'rlnMaskPickingRot': 'DOUBLE', 'rlnMaskPickingTilt': 'DOUBLE', 'rlnMaskPickingPSI': 'DOUBLE', 'rlnAutopickFigureOfMerit': 'DOUBLE', 'rlnHelicalTubeID': 'INT', 'rlnHelicalTubePitch': 'DOUBLE', 'rlnHelicalTrackLength': 'DOUBLE', 'rlnHelicalTrackLengthAngst': 'DOUBLE', 'rlnClassNumber': 'INT', 'rlnLogLikeliContribution': 'DOUBLE', 'rlnParticleId': 'INT', 'rlnParticleFigureOfMerit': 'DOUBLE', 'rlnKullbackLeiblerDivergence': 'DOUBLE', 'rlnRandomSubset': 'INT', 'rlnBeamTiltClass': 'INT', 'rlnParticleName': 'STRING', 'rlnOriginalParticleName': 'STRING', 'rlnNrOfSignificantSamples': 'INT', 'rlnNrOfFrames': 'INT', 'rlnAverageNrOfFrames': 'INT', 'rlnMovieFramesRunningAverage': 'INT', 'rlnMaxValueProbDistribution': 'DOUBLE', 'rlnParticleNumber': 'INT', 'rlnPipeLineJobCounter': 'INT', 'rlnPipeLineNodeType': 'INT', 'rlnPipeLineProcessType': 'INT', 'rlnPipeLineProcessStatus': 'INT', 'rlnFinalResolution': 'DOUBLE', 'rlnBfactorUsedForSharpening': 'DOUBLE', 'rlnParticleBoxFractionMolecularWeight': 'DOUBLE', 'rlnParticleBoxFractionSolventMask': 'DOUBLE', 'rlnFourierShellCorrelation': 'DOUBLE', 'rlnFourierShellCorrelationCorrected': 'DOUBLE', 'rlnFourierShellCorrelationParticleMolWeight': 'DOUBLE', 'rlnFourierShellCorrelationParticleMaskFraction': 'DOUBLE', 'rlnFourierShellCorrelationMaskedMaps': 'DOUBLE', 'rlnFourierShellCorrelationUnmaskedMaps': 'DOUBLE', 'rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps': 'DOUBLE', 'rlnAmplitudeCorrelationMaskedMaps': 'DOUBLE', 'rlnAmplitudeCorrelationUnmaskedMaps': 'DOUBLE', 'rlnDifferentialPhaseResidualMaskedMaps': 'DOUBLE', 'rlnDifferentialPhaseResidualUnmaskedMaps': 'DOUBLE', 'rlnFittedInterceptGuinierPlot': 'DOUBLE', 'rlnFittedSlopeGuinierPlot': 'DOUBLE', 'rlnCorrelationFitGuinierPlot': 'DOUBLE', 'rlnLogAmplitudesOriginal': 'DOUBLE', 'rlnLogAmplitudesMTFCorrected': 'DOUBLE', 'rlnLogAmplitudesWeighted': 'DOUBLE', 'rlnLogAmplitudesSharpened': 'DOUBLE', 'rlnLogAmplitudesIntercept': 'DOUBLE', 'rlnResolutionSquared': 'DOUBLE', 'rlnMolecularWeight': 'DOUBLE', 'rlnMtfValue': 'DOUBLE', 'rlnRandomiseFrom': 'DOUBLE', 'rlnUnfilteredMapHalf1': 'STRING', 'rlnUnfilteredMapHalf2': 'STRING', 'rlnIs3DSampling': 'BOOL', 'rlnIs3DTranslationalSampling': 'BOOL', 'rlnHealpixOrder': 'INT', 'rlnHealpixOrderOriginal': 'INT', 'rlnTiltAngleLimit': 'DOUBLE', 'rlnOffsetRange': 'DOUBLE', 'rlnOffsetStep': 'DOUBLE', 'rlnOffsetRangeOriginal': 'DOUBLE', 'rlnOffsetStepOriginal': 'DOUBLE', 'rlnHelicalOffsetStep': 'DOUBLE', 'rlnSamplingPerturbInstance': 'DOUBLE', 'rlnSamplingPerturbFactor': 'DOUBLE', 'rlnPsiStep': 'DOUBLE', 'rlnPsiStepOriginal': 'DOUBLE', 'rlnSymmetryGroup': 'STRING', 'rlnScheduleEdgeNumber': 'INT', 'rlnScheduleEdgeInputNodeName': 'STRING', 'rlnScheduleEdgeOutputNodeName': 'STRING', 'rlnScheduleEdgeIsFork': 'BOOL', 'rlnScheduleEdgeOutputNodeNameIfTrue': 'STRING', 'rlnScheduleEdgeBooleanVariable': 'STRING', 'rlnScheduleCurrentNodeName': 'STRING', 'rlnScheduleOriginalStartNodeName': 'STRING', 'rlnScheduleEmailAddress': 'STRING', 'rlnScheduleName': 'STRING', 'rlnScheduleJobName': 'STRING', 'rlnScheduleJobNameOriginal': 'STRING', 'rlnScheduleJobMode': 'STRING', 'rlnScheduleJobHasStarted': 'BOOL', 'rlnScheduleOperatorName': 'STRING', 'rlnScheduleOperatorType': 'STRING', 'rlnScheduleOperatorInput1': 'STRING', 'rlnScheduleOperatorInput2': 'STRING', 'rlnScheduleOperatorOutput': 'STRING', 'rlnScheduleBooleanVariableName': 'STRING', 'rlnScheduleBooleanVariableValue': 'BOOL', 'rlnScheduleBooleanVariableResetValue': 'BOOL', 'rlnScheduleFloatVariableName': 'STRING', 'rlnScheduleFloatVariableValue': 'DOUBLE', 'rlnScheduleFloatVariableResetValue': 'DOUBLE', 'rlnScheduleStringVariableName': 'STRING', 'rlnScheduleStringVariableValue': 'STRING', 'rlnScheduleStringVariableResetValue': 'STRING', 'rlnSelected': 'INT', 'rlnParticleSelectZScore': 'DOUBLE', 'rlnSortedIndex': 'INT', 'rlnStarFileMovieParticles': 'STRING', 'rlnPerFrameCumulativeWeight': 'DOUBLE', 'rlnPerFrameRelativeWeight': 'DOUBLE', 'rlnResolution': 'DOUBLE', 'rlnAngstromResolution': 'DOUBLE', 'rlnResolutionInversePixel': 'DOUBLE', 'rlnSpectralIndex': 'INT', 'rlnUnknownLabel': 'UNKNOWN'}
PROGRAM = 'image_handler'
OPTIONS = [('--i', 'str', None, 'Input STAR file, image (.mrc) or movie/stack (.mrcs)'), ('--o', 'str', '', "Output name (for STAR-input: insert this string before each image's extension)"), ('--multiply_constant', 'float', '1', 'Multiply the image(s) pixel values by this constant'), ('--divide_constant', 'float', '1', 'Divide the image(s) pixel values by this constant'), ('--add_constant', 'float', '0.', 'Add this constant to the image(s) pixel values'), ('--subtract_constant', 'float', '0.', 'Subtract this constant from the image(s) pixel values'), ('--threshold_above', 'float', '999.', 'Set all values higher than this value to this value'), ('--threshold_below', 'float', '-999.', 'Set all values lower than this value to this value'), ('--multiply', 'str', '', 'Multiply input image(s) by the pixel values in this image'), ('--divide', 'str', '', 'Divide input image(s) by the pixel values in this image'), ('--add', 'str', '', 'Add the pixel values in this image to the input image(s) '), ('--subtract', 'str', '', 'Subtract the pixel values in this image to the input image(s) '), ('--fsc', 'str', '', 'Calculate FSC curve of the input image with this image'), ('--power', 'bool', None, 'Calculate power spectrum (|F|^2) of the input image'), ('--adjust_power', 'str', '', 'Adjust the power spectrum of the input image to be the same as this image '), ('--fourier_filter', 'str', '', 'Multiply the Fourier transform of the input image(s) with this one image '), ('--fsc_weighting', 'str', '', 'Multiply the Fourier transform of the input image(s) with this FSC curve'), ('--test_F', 'bool', None, 'Test FFT for input realspace image'), ('--C_OUT', 'int', '300', 'Output YSIZE of test_F'), ('--HistGram', 'str', '', 'Calculate histogram in value of input image.'), ('--optimise_scale_subtract', 'bool', None, 'Optimise scale between maps before subtraction?'), ('--optimise_bfactor_subtract', 'float', '0.', 'Search range for relative B-factor for subtraction (in A^2)'), ('--mask_optimise_subtract', 'str', '', 'Use only voxels in this mask to optimise scale for subtraction'), ('--gen_MP_table', 'bool', None, 'Read MRC file and generate Mask-Picking table. Use together with --MP_table and --MP_stepsize'), ('--MP_table', 'str', '', 'Output name of Mask-Picking table.'), ('--MP_stepsize', 'float', '0.9375', 'Stepsize of Mask-picking table'), ('--postprocess', 'bool', None, 'Do postprocess for single map.'), ('--post_fsc', 'str', '', 'FSC to be applied.'), ('--lp', 'float', '-1', 'Low-passed frequency'), ('--bf', 'float', '0.', 'B-factor to be applied'), ('--stats', 'bool', None, 'Calculate per-image statistics?'), ('--com', 'bool', None, 'Calculate center of mass?'), ('--bfactor', 'float', '0.', 'Apply a B-factor (in A^2)'), ('--lowpass', 'float', '-1.', 'Low-pass filter frequency (in A)'), ('--highpass', 'float', '-1.', 'High-pass filter frequency (in A)'), ('--directional', 'str', '', "Directionality of low-pass filter frequency ('X', 'Y' or 'Z', default non-directional)"), ('--LoG', 'float', '-1.', 'Diameter for optimal response of Laplacian of Gaussian filter (in A)'), ('--angpix', 'float', '-1', 'Pixel size (in A)'), ('--rescale_angpix', 'float', '-1.', 'Scale input image(s) to this new pixel size (in A)'), ('--meanshrink', 'float', '1.', 'Shrink the map down to the scale.'), ('--force_header_angpix', 'float', '-1.', 'Change the pixel size in the header (in A). Without --rescale_angpix, the image is not scaled.'), ('--new_box', 'int', '-1', 'Resize the image(s) to this new box size (in pixel) '), ('--filter_edge_width', 'int', '2', 'Width of the raised cosine on the low/high-pass filter edge (in resolution shells)'), ('--flipX', 'bool', None, 'Flip (mirror) a 2D image or 3D map in the X-direction?'), ('--flipY', 'bool', None, 'Flip (mirror) a 2D image or 3D map in the Y-direction?'), ('--flipZ', 'bool', None, 'Flip (mirror) a 3D map in the Z-direction?'), ('--invert_hand', 'bool', None, 'Invert hand by flipping X? Similar to flipX, but preserves the symmetry origin. Edge pixels are wrapped around.'), ('--shift_com', 'bool', None, 'Shift image(s) to their center-of-mass (only on positive pixel values)'), ('--shift_x', 'float', '0.', 'Shift images this many pixels in the X-direction'), ('--shift_y', 'float', '0.', 'Shift images this many pixels in the Y-direction'), ('--shift_z', 'float', '0.', 'Shift images this many pixels in the Z-direction'), ('--avg_ampl', 'bool', None, 'Calculate average amplitude spectrum for all images?'), ('--avg_ampl2', 'bool', None, 'Calculate average amplitude spectrum for all images?'), ('--avg_ampl2_ali', 'bool', None, 'Calculate average amplitude spectrum for all aligned images?'), ('--average', 'bool', None, 'Calculate average of all images (without alignment)'), ('--correct_avg_ampl', 'str', '', 'Correct all images with this average amplitude spectrum'), ('--minr_ampl_corr', 'int', '0', 'Minimum radius (in Fourier pixels) to apply average amplitudes'), ('--remove_nan', 'bool', None, 'Replace non-numerical values (NaN, inf, etc) in the image(s)'), ('--replace_nan', 'float', '0', 'Replace non-numerical values (NaN, inf, etc) with this value'), ('--phase_randomise', 'float', '-1', 'Randomise phases beyond this resolution (in Angstroms)'), ('--normal_background_powerspectrum', 'bool', None, 'Normalize background powerspectrum.'), ('--bg_radius', 'int', '64.', 'background radius in pixels, combined with --normal_background_powerspectrum.'), ('--boxsize', 'int', '128', 'The boxsize of input particles, combined with --normal_background_powerspectrum.'), ('--sym', 'str', '', 'Symmetrise 3D map with this point group (e.g. D6)'), ('--flipXY', 'bool', None, 'Flip the image(s) in the XY direction?'), ('--flipmXY', 'bool', None, 'Flip the image(s) in the -XY direction?'), ('--add_edge', 'bool', None, 'Add a barcode-like edge to the micrograph/movie frames?'), ('--edge_x0', 'int', '0', 'Pixel column to be used for the left edge'), ('--edge_y0', 'int', '0', 'Pixel row to be used for the top edge'), ('--edge_xF', 'int', '4095', 'Pixel column to be used for the right edge'), ('--edge_yF', 'int', '4095', 'Pixel row to be used for the bottom edge'), ('--avg_bin', 'int', '-1', 'Width (in frames) for binning average, i.e. of every so-many frames'), ('--avg_first', 'int', '-1', 'First frame to include in averaging'), ('--avg_last', 'int', '-1', 'Last frame to include in averaging'), ('--average_all_movie_frames', 'bool', None, 'Average all movie frames of all movies in the input STAR file.'), ('--black', 'float', '0', 'Pixel value for black (default is auto-contrast)'), ('--white', 'float', '0', 'Pixel value for white (default is auto-contrast)'), ('--sigma_contrast', 'float', '0', 'Set white and black pixel values this many times the image stddev from the mean'), ('--colour_fire', 'bool', None, 'Show images in black-grey-white-red colour scheme (highlight high signal)?'), ('--colour_ice', 'bool', None, 'Show images in blue-black-grey-white colour scheme (highlight low signal)?'), ('--colour_fire-n-ice', 'bool', None, 'Show images in blue-grey-red colour scheme (highlight high&low signal)?'), ('--colour_rainbow', 'bool', None, 'Show images in cyan-blue-black-red-yellow colour scheme?'), ('--colour_difference', 'bool', None, 'Show images in cyan-blue-black-red-yellow colour scheme (for difference images)?'), ('--cos_dphi', 'str', '', 'Shell mean cosine of phase differences')]
import argparse
import copy
import glob
import math
import os
from pathlib import Path
import random
import re
import sys

VERSION = '1.0.0'

def cli():
    p = argparse.ArgumentParser(description=f'Standalone Python {PROGRAM}, RELION 3.1.2 SIRM v1p2 port',
                                allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for flag, typ, default, helptext in OPTIONS:
        kw = dict(dest=flag[2:].replace('-','_'),help=helptext)
        if typ == 'bool': kw.update(action='store_true')
        else:
            convert = {'int':lambda x:int(float(x)), 'float':float, 'str':str}[typ]
            kw.update(type=convert)
            if default is None: kw.update(required=True)
            else: kw.update(default=convert(default))
        p.add_argument(flag,**kw)
    p.add_argument('--version',action='version',version=f'{PROGRAM} Python {VERSION}')
    p.add_argument('--data_root',default='',help='Root for relative paths referenced by input STAR files')
    if not any(x[0]=='--random_seed' for x in OPTIONS):
        p.add_argument('--random_seed',type=int,default=0,help='Python random seed; negative for random seed')
    return p.parse_args()

def fail(message):
    raise ValueError(message)

def warn(message):
    print('WARNING: '+message,file=sys.stderr)

def label(s):
    return s if s.startswith('_') else '_'+s

def fmt(x):
    return format(float(x),'.12g')

def suffix(path,tag):
    p=Path(path)
    return str(p.with_name(p.stem+tag+p.suffix))

class Token:
    def __init__(self,value,quoted=False): self.value,self.quoted=value,quoted

def star_tokens(text):
    """STAR lexer: quotes, comments, multiline semicolon fields, no backslash escaping."""
    lines=iter(text.splitlines())
    for line in lines:
        if line.startswith(';'):
            values=[line[1:]]
            for line in lines:
                if line.startswith(';'): break
                values.append(line)
            else: fail('Unterminated STAR semicolon field')
            yield Token('\n'.join(values),True)
            continue
        i=0
        while i<len(line):
            if line[i].isspace(): i+=1; continue
            if line[i]=='#': break
            if line[i] in "\"'":
                q=line[i]; start=i=i+1
                while i<len(line) and not (line[i]==q and (i+1==len(line) or line[i+1].isspace() or line[i+1]=='#')): i+=1
                if i==len(line): fail('Unterminated STAR quote: '+line)
                yield Token(line[start:i],True); i+=1
            else:
                start=i
                while i<len(line) and not line[i].isspace(): i+=1
                yield Token(line[start:i])

def star_special(t):
    return not t.quoted and (t.value.startswith('_') or t.value.lower().startswith(('data_','save_')) or t.value.lower() in ('loop_','stop_','global_'))

class Table:
    def __init__(self,labels=None,rows=None):
        self.labels=list(labels or []); self.rows=list(rows or [])
    def col(self,key):
        key=label(key)
        if key not in self.labels: fail('Missing STAR column '+key)
        return self.labels.index(key)
    def add(self,key,value='?'):
        key=label(key)
        if key not in self.labels:
            self.labels.append(key)
            for row in self.rows: row.append(str(value))
        return self.col(key)
    def get(self,row,key,default=None):
        if label(key) not in self.labels:
            if default is not None: return default
            fail('Missing STAR column '+label(key))
        return row[self.col(key)]

class Block:
    def __init__(self,name): self.name=name; self.items=[]

class Star:
    def __init__(self,blocks=None): self.blocks=list(blocks or [])
    @classmethod
    def read(cls,path):
        tokens=list(star_tokens(Path(path).read_text(encoding='utf-8-sig')))
        result=cls(); block=None; i=0
        while i<len(tokens):
            t=tokens[i]; v=t.value; lo=v.lower()
            if not t.quoted and lo.startswith('data_'):
                block=Block(v[5:]); result.blocks.append(block); i+=1
            elif block is None: fail('STAR content before data_ block')
            elif lo=='loop_' and not t.quoted:
                i+=1; labs=[]
                while i<len(tokens) and tokens[i].value.startswith('_') and not tokens[i].quoted:
                    labs.append(tokens[i].value); i+=1
                if not labs or len(labs)!=len(set(labs)): fail('Invalid/duplicate STAR loop labels')
                rows=[]; row=[]
                while i<len(tokens) and not (not row and star_special(tokens[i])):
                    if star_special(tokens[i]): fail('Incomplete STAR loop row before '+tokens[i].value)
                    row.append(tokens[i].value); i+=1
                    if len(row)==len(labs): rows.append(row); row=[]
                if row: fail('Incomplete STAR loop row')
                block.items.append(Table(labs,rows))
                if i<len(tokens) and tokens[i].value.lower()=='stop_': i+=1
            elif v.startswith('_') and not t.quoted:
                if i+1>=len(tokens) or star_special(tokens[i+1]): fail('Missing STAR scalar value '+v)
                block.items.append((v,tokens[i+1].value)); i+=2
            else: fail('Unsupported STAR token '+v)
        return result
    def tables(self):
        for b in self.blocks:
            for t in b.items:
                if isinstance(t,Table): yield b,t
    def table(self,name=None):
        pairs=list(self.tables())
        if name and name!='discover':
            matches=[(b,t) for b,t in pairs if b.name==name.removeprefix('data_')]
        else:
            matches=[(b,t) for b,t in pairs if b.name in ('particles','micrographs','movies')]
            if not matches: matches=[(b,t) for b,t in pairs if b.name!='optics']
        if not matches: fail('No matching STAR loop: '+str(name))
        return matches[0]
    def text(self):
        out=['# version 30001','']
        for b in self.blocks:
            out+=['data_'+b.name,'']
            for item in b.items:
                if isinstance(item,Table):
                    out+=['loop_']+[f'{x} #{i+1}' for i,x in enumerate(item.labels)]
                    for row in item.rows:
                        vals=[star_quote(str(v)) for v in row]
                        out.append('\n'.join(vals) if any('\n' in v for v in vals) else ' '.join(vals))
                else:
                    k,v=item; q=star_quote(str(v)); out.append(k+('\n' if '\n' in q else ' ')+q)
                out.append('')
        return '\n'.join(out)+'\n'
    def write(self,path):
        import tempfile
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.'+Path(path).name+'.',suffix='.tmp',dir=Path(path).resolve().parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as stream: stream.write(self.text())
            os.replace(tmp,path)
        finally:
            if Path(tmp).exists(): Path(tmp).unlink()

def star_quote(v):
    if '\n' in v or ('"' in v and "'" in v):
        if any(x.startswith(';') for x in v.splitlines()): fail('STAR multiline value contains a semicolon at line start')
        return ';'+v+'\n;'
    if not v or any(c.isspace() for c in v) or v[0] in '_#;\"\'' or v.lower().startswith(('data_','save_')) or v.lower() in ('loop_','stop_','global_'):
        q="'" if "'" not in v else '"'
        return q+v+q
    return v

def star_table(name,labels,rows):
    b=Block(name); b.items.append(Table([label(x) for x in labels],[[str(v) for v in r] for r in rows])); return Star([b])

def resolve_image(name,starpath=None,data_root=''):
    prefix=''
    if '@' in name:
        idx,name=name.split('@',1); prefix=idx+'@'
    p=Path(name)
    if p.is_absolute() and p.exists(): return prefix+str(p)
    candidates=[]
    if data_root: candidates.append(Path(data_root)/p)
    candidates.append(p)
    if starpath:
        # RELION paths are relative to project root, which can be a parent of the STAR.
        candidates.extend(parent/p for parent in Path(starpath).resolve().parents)
    for candidate in candidates:
        if candidate.exists(): return prefix+str(candidate.resolve())
    fail('Cannot resolve image '+name+'; use --data_root for the RELION project root')

def pixel_for(star,table,row,fallback=1):
    if '_rlnImagePixelSize' in table.labels: return float(table.get(row,'rlnImagePixelSize'))
    for b,t in star.tables():
        if b.name=='optics':
            gid=table.get(row,'rlnOpticsGroup')
            for r in t.rows:
                if t.get(r,'rlnOpticsGroup')==gid:
                    for key in ('rlnImagePixelSize','rlnMicrographPixelSize','rlnMicrographOriginalPixelSize'):
                        if label(key) in t.labels: return float(t.get(r,key))
                    return fallback
            fail('Particle refers to missing optics group '+gid)
    return fallback

def entrypoint():
    try: main(cli())
    except (ValueError,OSError,ImportError,IndexError,KeyError) as e:
        print('ERROR: '+str(e),file=sys.stderr); sys.exit(1)


# Heavy libraries are loaded only for operations that need them.
def numerical(scipy=False,mrc=True):
    global np, mrcfile, ndi
    import numpy as np
    if mrc: import mrcfile
    if scipy:
        from scipy import ndimage as ndi

def read_mrc(name):
    index=None
    if '@' in str(name):
        s,name=str(name).split('@',1); index=int(s)-1
        if index<0: fail('MRC stack indices are one based')
    with mrcfile.mmap(name,permissive=False,mode='r') as m:
        if tuple(int(m.header[k]) for k in ('mapc','mapr','maps'))!=(1,2,3):
            fail('Nonstandard MRC axis mapping; reorder axes before processing')
        ap=float(m.voxel_size.x)
        if ap<=0: ap=1.0
        a=m.data
        if index is not None:
            if a.ndim==2 and index==0: pass
            elif a.ndim<3 or index>=a.shape[0]: fail('MRC stack index out of range')
            else: a=a[index]
        meta={'angpix':ap,'origin':tuple(float(m.header.origin[k]) for k in ('x','y','z')),
              'is_stack':bool(m.is_image_stack() or str(name).lower().endswith('.mrcs'))}
        return np.array(a,dtype=np.float64),meta

def write_mrc(path,a,ap=1.0,meta=None,stack=False):
    if '@' in str(path):
        number,path=str(path).split('@',1); index=int(number)-1
        if index<0 or a.ndim!=2: fail('Indexed stack output requires a positive one-based index and a 2D image')
        if not Path(path).exists():
            if index!=0: fail('For a new indexed stack, start with 1@filename.mrcs')
            return write_mrc(path,a[None,...],ap,meta,True)
        with mrcfile.mmap(path,mode='r') as old:
            shape=old.data.shape; count=shape[0] if len(shape)==3 else 1
            if shape[-2:]!=a.shape: fail('Indexed output shape differs from existing stack')
            oldap=float(old.voxel_size.x)
            if oldap>0 and not math.isclose(oldap,ap,rel_tol=1e-6): fail('Indexed output pixel size differs from existing stack')
        if index<count:
            with mrcfile.mmap(path,mode='r+') as m:
                if m.data.ndim==2: m.data[:]=a
                else: m.data[index]=a
                m.update_header_stats()
            return
        if index!=count: fail('Indexed output may replace a frame or append the next frame; gaps are not allowed')
        import tempfile
        fd,tmp=tempfile.mkstemp(suffix='.mrcs',dir=Path(path).resolve().parent); os.close(fd)
        try:
            with mrcfile.mmap(path,mode='r') as old,mrcfile.new_mmap(tmp,shape=(count+1,*a.shape),mrc_mode=2,overwrite=True) as out:
                out.set_image_stack(); out.voxel_size=ap
                for i in range(count): out.data[i]=old.data[i] if old.data.ndim==3 else old.data
                out.data[count]=a; out.update_header_stats()
            os.replace(tmp,path)
        finally:
            if Path(tmp).exists(): Path(tmp).unlink()
        return
    if Path(path).suffix.lower() not in ('.mrc','.mrcs','.map'): fail('Output must be .mrc, .mrcs, .map or explicitly supported .png')
    if not np.isfinite(ap) or ap<=0: fail('Pixel size must be positive')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with mrcfile.new(path,overwrite=True) as m:
        m.set_data(np.asarray(a,dtype=np.float32))
        if stack and a.ndim==3: m.set_image_stack()
        elif a.ndim==3: m.set_volume()
        m.voxel_size=ap
        if meta: m.header.origin=meta.get('origin',(0.,0.,0.))
        m.update_header_stats()

def round_rel(x):
    return np.copysign(np.floor(np.abs(x)+0.5),x).astype(np.int64)

def grids(shape,fourier=False):
    vec=[]
    for i,n in enumerate(shape):
        if fourier:
            v=np.arange(n//2+1,dtype=float) if i==len(shape)-1 else np.rint(np.fft.fftfreq(n)*n)
            if i<len(shape)-1 and n%2==0: v[n//2]=n//2
        else: v=np.arange(n,dtype=float)-n//2
        sh=[1]*len(shape); sh[i]=len(v); vec.append(v.reshape(sh))
    return vec

def radius(shape,fourier=False):
    return np.sqrt(sum(g*g for g in grids(shape,fourier)))

def lowpass_weights(shape,ap,res,width=2,direction=''):
    if width<0 or res<=0 or ap<=0: fail('Invalid filter width, resolution or pixel size')
    n=shape[-1]; r=radius(shape,True)
    if direction:
        direction=direction.upper()
        if direction not in ('X','Y','Z') or ('XYZ'.index(direction)>=len(shape)): fail('Invalid filter direction')
        r=np.abs(grids(shape,True)[-1-'XYZ'.index(direction)])
    cutoff=int(math.floor(n*ap/res+0.5)); half=width//2
    lo=max(0,cutoff-half); hi=cutoff+half
    if hi<=lo: return (r<cutoff).astype(float)
    return 0.5+0.5*np.cos(np.pi*np.clip((r-lo)/(hi-lo),0,1))

def b_weights(shape,ap,b):
    r=radius(shape,True); s=r/(shape[-1]*ap)
    with np.errstate(over='raise'):
        return np.where(r<=shape[-1]/2,np.exp(-b*s*s/4),0)

def filter_map(a,ap,res,width=2,high=False,direction='',rng=None):
    oldshape=a.shape
    if not high and len(set(a.shape))!=1:
        if a.ndim!=2: fail('Filtering requires a cubic 3D map')
        a=pad_noise(a,rng or RelionRNG(0))
    w=lowpass_weights(a.shape,ap,res,width,direction)
    if high: w=1-w
    result=np.fft.irfftn(np.fft.rfftn(a)*w,s=a.shape)
    return window(result,oldshape) if result.shape!=oldshape else result

def shell_sum(values,shell,n):
    return np.bincount(shell.ravel(),weights=np.asarray(values).ravel(),minlength=n)[:n]

def fsc(a,b,selection=None):
    if a.shape!=b.shape: fail('FSC input shapes differ')
    f=np.fft.rfftn(a); g=np.fft.rfftn(b)
    return fsc_fourier(f,g,a.shape,selection)

def fsc_fourier(f,g,shape,selection=None):
    shell=round_rel(radius(shape,True)); n=shape[-1]//2+1
    w=1. if selection is None else selection
    num=shell_sum((f*g.conj()).real*w,shell,n)
    den=np.sqrt(shell_sum(np.abs(f)**2*w,shell,n)*shell_sum(np.abs(g)**2*w,shell,n))
    return np.divide(num,den,out=np.zeros(n),where=den>0)

def shell_mean(values,shape,n=None):
    shell=round_rel(radius(shape,True)); n=n or shape[-1]//2+1
    c=shell_sum(np.ones_like(values,dtype=float),shell,n)
    return shell_sum(values,shell,n)/np.maximum(c,1)

def radial_values(curve,shape,outside=0):
    s=round_rel(radius(shape,True)); return np.where(s<len(curve),curve[np.minimum(s,len(curve)-1)],outside)

def fsc_weights(curve,shape):
    c=np.asarray(curve).copy(); bad=np.flatnonzero(c<0.0001)
    if len(bad): c[bad[0]:]=0
    return radial_values(np.sqrt(2*np.maximum(c,0)/(1+np.maximum(c,0))),shape)

class RelionRNG:
    """Portable glibc rand() stream, matching the supplied Linux CPU executable."""
    def __init__(self,seed=0):
        import time
        seed=int(time.time()) if seed<0 else seed
        seed=(seed & 0xffffffff) or 1
        state=[seed]
        for i in range(1,31):
            prev=state[-1]; prev=prev if prev<2147483648 else prev-4294967296
            state.append((16807*prev)%2147483647)
        state+=state[:3]
        for i in range(34,344): state.append((state[i-31]+state[i-3])&0xffffffff)
        self.state=state[-31:]; self.pos=0; self.pending=np.empty(0,dtype=np.uint32); self.gaussian_spare=None
    def raw(self,count):
        prefix=self.pending[:count]; self.pending=self.pending[len(prefix):]
        count-=len(prefix)
        def stream():
            st=self.state; p=self.pos
            for _ in range(count):
                v=(st[p]+st[(p+28)%31])&0xffffffff; st[p]=v; p=(p+1)%31
                yield v>>1
        values=np.fromiter(stream(),dtype=np.uint32,count=count)
        self.pos=(self.pos+count)%31
        return np.concatenate((prefix,values)) if len(prefix) else values
    def uniform(self,lo,hi,size):
        count=int(np.prod(size)) if not isinstance(size,int) else size
        values=self.raw(count).astype(np.float32)
        # rnd_unif arguments and its result are float, even in double precision RELION.
        span=np.float32(hi)-np.float32(lo); denom=np.float32(np.float32(2147483647)/span)
        result=np.float32(lo)+values/denom
        return result.reshape(size).astype(float)
    def normal(self,mu,sigma,count):
        mu=np.float32(mu); sigma=np.float32(sigma)
        if sigma==0: return np.full(count,mu,dtype=float)
        out=np.empty(count,dtype=np.float32); pos=0
        if self.gaussian_spare is not None and count:
            out[0]=mu+sigma*self.gaussian_spare; pos=1; self.gaussian_spare=None
        while pos<count:
            pairs=min(65536,max(1,(count-pos+1)//2)); raw=self.raw(pairs*2).reshape(-1,2)
            u=np.float32(-1)+raw.astype(np.float32)/np.float32(2147483647)*np.float32(2)
            w=np.sum(u.astype(float)**2,axis=1).astype(np.float32); valid=np.flatnonzero((w>0)&(w<1))
            if not len(valid): continue
            take=min(len(valid),(count-pos+1)//2); last=valid[take-1]+1
            if last<len(raw): self.pending=np.concatenate((raw[last:].ravel(),self.pending))
            selected=valid[:take]; ww=w[selected].astype(float)
            mult=np.sqrt(-2*np.log(ww)/ww).astype(np.float32)
            values=(u[selected]*mult[:,None]).ravel(); needed=min(count-pos,len(values))
            out[pos:pos+needed]=mu+sigma*values[:needed]; pos+=needed
            if needed<len(values): self.gaussian_spare=values[-1]
        return out.astype(float)

def pad_noise(x,rng):
    n=max(x.shape); padded=window(x,n); present=window(np.ones(x.shape,dtype=bool),n)
    padded[~present]=rng.normal(x.mean(),x.std(ddof=0),int(np.count_nonzero(~present)))
    return padded

def random_phases(a,index,rng):
    f=np.fft.rfftn(a); selected=radius(a.shape,True)>=index
    phase=rng.uniform(0,2*np.pi,int(np.count_nonzero(selected)))
    # Same half-complex convention as RELION; inverse FFT makes a real map.
    f[selected]=np.abs(f[selected])*np.exp(1j*phase)
    return np.fft.irfftn(f,s=a.shape)

def window(a,shape):
    if isinstance(shape,int): shape=(shape,)*a.ndim
    if len(shape)!=a.ndim or min(shape)<1: fail('Invalid output box')
    out=np.zeros(shape,dtype=a.dtype); src=[]; dst=[]
    for old,new in zip(a.shape,shape):
        lo=max(-old//2 if old%2==0 else -(old//2),-(new//2))
        hi=min(old-old//2,new-new//2)
        src.append(slice(lo+old//2,hi+old//2)); dst.append(slice(lo+new//2,hi+new//2))
    out[tuple(dst)]=a[tuple(src)]; return out

def resize_fourier(a,n):
    if len(set(a.shape))!=1: fail('Fourier resizing requires a square or cubic box')
    old=a.shape[-1]
    if n==old: return a.copy()
    f=np.fft.rfftn(a)/a.size
    shape=(n,)*a.ndim; target=np.zeros(shape[:-1]+(n//2+1,),dtype=complex)
    coords=grids(shape,True)
    # RELION windowFourierTransform copies half-complex samples without Nyquist splitting.
    idx=[]; valid=np.ones(target.shape,dtype=bool)
    for k,(g,oldn) in enumerate(zip(coords,a.shape)):
        if k==a.ndim-1: valid &= g<=oldn//2; idx.append(np.minimum(g,oldn//2).astype(int))
        else:
            valid &= np.abs(g)<=oldn//2
            if n>oldn and oldn%2==0: valid &= g!=-oldn//2
            idx.append(np.mod(g,oldn).astype(int))
    if n>old: valid &= radius(shape,True)<=old//2
    target[valid]=np.broadcast_to(f[tuple(idx)],target.shape)[valid]
    return np.fft.irfftn(target,s=shape)*np.prod(shape)

def euler_matrix(rot,tilt,psi):
    a,b,g=np.deg2rad([rot,tilt,psi]); ca,sa=np.cos(a),np.sin(a); cb,sb=np.cos(b),np.sin(b); cg,sg=np.cos(g),np.sin(g)
    return np.array([[cg*cb*ca-sg*sa,cg*cb*sa+sg*ca,-cg*sb],[-sg*cb*ca-cg*sa,-sg*cb*sa+cg*ca,sg*sb],[sb*ca,sb*sa,cb]])

def matrix_euler(m):
    ab=np.hypot(m[0,2],m[1,2])
    if ab>1e-10:
        a=np.arctan2(m[2,1],m[2,0]); g=np.arctan2(m[1,2],-m[0,2]); b=np.arctan2(ab,m[2,2])
    else:
        a=0.; b=0. if m[2,2]>0 else np.pi
        g=np.arctan2(-m[1,0],m[0,0]) if b==0 else np.arctan2(m[1,0],-m[0,0])
    return np.rad2deg([a,b,g])

def symmetry_operators(sym):
    name=sym or 'C1'; sym=name.upper(); identity=np.eye(3); lh=np.diag([1.,1.,-1.])
    rot=[]; mirrors=[]; inversion=False
    match=re.fullmatch('([CDS])([1-9][0-9]*)([VH]?)',sym)
    if match:
        kind,num,tail=match.groups(); num=int(num)
        if kind=='S':
            if num%2: fail('S symmetry order must be even')
            num//=2; inversion=True
        rot=[(num,[0,0,1])]
        if kind=='D': rot.append((2,[1,0,0]))
        if tail: mirrors.append([0,0,1] if tail=='H' else [1,0,0] if kind=='D' else [0,1,0])
    elif sym in ('CI','CS'):
        inversion=sym=='CI'; mirrors=[] if inversion else [[0,0,1]]
    elif sym in ('T','TD','TH'):
        rot=[(3,[0,0,1]),(2,[0,.816496,.577350])]; inversion=sym=='TH'
        if sym=='TD': mirrors=[[1.4142136,2.4494897,0]]
    elif sym in ('O','OH'):
        rot=[(3,[.5773502]*3),(4,[0,0,1])]
        if sym=='OH': mirrors=[[0,1,1]]
    elif sym in ('I','IH','I1','I2','I3','I4','I1H','I2H','I3H','I4H'):
        conv=sym.rstrip('H'); conv='I2' if conv=='I' else conv
        axes={
            'I1':[(2,[1,0,0]),(5,[.85065080702670,0,-.5257311142635]),(3,[.9341723640,.3568220765,0])],
            'I2':[(2,[0,0,1]),(5,[.525731114,0,.850650807]),(3,[0,.356822076,.934172364])],
            'I3':[(2,[-.5257311143,0,.8506508070]),(5,[0,0,1]),(3,[-.4911234778630044,.3568220764705179,.7946544753759428])],
            'I4':[(2,[.5257311143,0,.8506508070]),(5,[.8944271932547096,0,.4472135909903704]),(3,[.4911234778630044,.3568220764705179,.7946544753759428])]}
        rot=axes[conv]
        if sym.endswith('H'): mirrors=[{'I1':[0,0,-1],'I2':[1,0,0],'I3':[.850650807,0,.525731114],'I4':[.850650807,0,-.525731114]}[conv]]
    elif Path(name).is_file():
        for line in Path(name).read_text().splitlines():
            words=line.split('#',1)[0].replace(',',' ').split()
            if not words: continue
            if words[0]=='rot_axis': rot.append((int(words[1]),list(map(float,words[2:5]))))
            elif words[0]=='mirror_plane': mirrors.append(list(map(float,words[1:4])))
            elif words[0]=='inversion': inversion=True
            else: fail('Unknown symmetry directive '+words[0])
    else: fail('Unsupported point group '+sym)
    ops=[]
    for fold,axis in rot:
        v=np.array(axis,float); v/=np.linalg.norm(v); k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        for j in range(1,fold):
            angle=np.deg2rad(360*j/fold); r=(identity+np.sin(angle)*k+(1-np.cos(angle))*(k@k)).T
            r[np.abs(r)<1e-6]=0; ops.append((identity.copy(),r))
    if inversion: ops.append((lh.copy(),-identity))
    for axis in mirrors:
        v=np.array(axis,float); v/=np.linalg.norm(v); ops.append((lh.copy(),identity-2*np.outer(v,v)))
    # Diagonal traversal is the upstream closure order, important for BBR row order.
    n=0
    while n<len(ops):
        for j in range(n,-1,-1):
            for i in range(n+1):
                if max(i,j)!=n: continue
                l=ops[i][0]@ops[j][0]; r=ops[i][1]@ops[j][1]
                if np.allclose(l,identity,atol=1e-5,rtol=0) and np.allclose(r,identity,atol=1e-5,rtol=0): continue
                if not any(np.allclose(l,ll,atol=1e-5,rtol=0) and np.allclose(r,rr,atol=1e-5,rtol=0) for ll,rr in ops):
                    l[np.abs(l)<1e-6]=0; r[np.abs(r)<1e-6]=0; ops.append((l,r))
                    if len(ops)>500: fail('Symmetry generators do not close to a supported finite group')
        n+=1
    return [(identity,identity)]+ops

def symmetry_matrices(sym): return [r for l,r in symmetry_operators(sym)]

def symmetrize(x,sym):
    mats=symmetry_matrices(sym); out=x.copy(); center=np.array(x.shape)//2
    for mat in mats[1:]:
        m=mat[::-1,::-1]
        for start in range(0,x.shape[0],16):
            stop=min(start+16,x.shape[0]); coords=np.indices((stop-start,*x.shape[1:]),dtype=float)
            coords[0]+=start; coords-=center[:,None,None,None]
            shape=coords.shape; src=(m@coords.reshape(3,-1)+center[:,None]).reshape(shape)
            for k,n in enumerate(x.shape):
                src[k]=np.where((src[k]>=-1e-6)&(src[k]<=n-1+1e-6),np.clip(src[k],0,n-1),src[k])
            out[start:stop]+=ndi.map_coordinates(x,src,order=1,mode='constant',cval=0,prefilter=False)
    return out/len(mats)

def automask(a,threshold,extend,width):
    if width<0: fail('Soft edge width cannot be negative')
    mask=a>=threshold
    if not mask.any(): return np.zeros(a.shape)
    if not mask.all() and extend:
        # Only in-box neighbors count in RELION, including at the box boundary.
        if extend>0: mask |= ndi.distance_transform_edt(~mask)<extend
        else:
            size=abs(math.ceil(extend)); rr=np.indices((2*size+1,)*a.ndim)-size
            structure=np.sum(rr*rr,axis=0)<extend*extend
            mask=ndi.binary_erosion(mask,structure=structure,border_value=1)
    if not mask.any(): return np.zeros(a.shape)
    if width>0 and not mask.all():
        d=ndi.distance_transform_edt(~mask); return 0.5+0.5*np.cos(np.pi*np.minimum(d/width,1))
    return mask.astype(float)

def spherical_mask(shape,inner,outer,width,center=None):
    g=grids(shape); center=center or [0]*len(shape)
    r=np.sqrt(sum((v-c)**2 for v,c in zip(g,center)))
    if width==0: return ((r>=inner)&(r<outer)).astype(float)
    v=np.ones(shape)
    v=np.where(r<inner,0.5-0.5*np.cos(np.pi*np.clip((r-inner+width)/width,0,1)),v)
    return np.where(r>=outer,0.5+0.5*np.cos(np.pi*np.clip((r-outer)/width,0,1)),v)

def curve_table(curves,n,ap,name='fsc'):
    rows=[]
    for i in range(len(next(iter(curves.values())))):
        res=n*ap/i if i else 999.
        rows.append([i,fmt(1/res),fmt(res)]+[fmt(c[i]) for c in curves.values()])
    return star_table(name,['rlnSpectralIndex','rlnResolution','rlnAngstromResolution']+list(curves),rows)


def read_curve(path):
    s=Star.read(path)
    choices=('rlnGoldStandardFsc','rlnFourierShellCorrelationCorrected','rlnFourierShellCorrelation','rlnFourierShellCorrelationUnmaskedMaps')
    for _,t in s.tables():
        for key in choices:
            if label(key) in t.labels:
                return np.array([float(t.get(r,key)) for r in t.rows])
    fail('No FSC column found in '+path)

def transform_image(x,ap,a,psi=0):
    src=x.copy(); y=x.copy()
    if a.add_edge:
        if x.ndim!=2: fail('--add_edge requires 2D images')
        if not (0<=a.edge_x0<=a.edge_xF<x.shape[1] and 0<=a.edge_y0<=a.edge_yF<x.shape[0]): fail('Barcode edge coordinates outside image')
        src=src[np.ix_(np.clip(np.arange(x.shape[0]),a.edge_y0,a.edge_yF),np.clip(np.arange(x.shape[1]),a.edge_x0,a.edge_xF))]; y=src.copy()
    if a.flipXY or a.flipmXY:
        if x.ndim!=2: fail('XY transpose requires a 2D image')
        y=src.T.copy() if a.flipXY else src[::-1,::-1].T.copy()
    if a.postprocess:
        if not a.post_fsc: fail('--postprocess requires --post_fsc')
        c=read_curve(a.post_fsc); c=np.sqrt(2*np.maximum(c,0)/(1+np.maximum(c,0)))
        f=np.fft.rfftn(src)*radial_values(c,src.shape)
        if a.lp>0: f*=lowpass_weights(src.shape,ap,a.lp)
        if a.bf: f*=b_weights(src.shape,ap,a.bf)
        y=np.fft.irfftn(f,s=src.shape)
    if a.remove_nan: y=np.nan_to_num(y,nan=a.replace_nan,posinf=a.replace_nan,neginf=a.replace_nan)
    if a.phase_randomise>0:
        y=random_phases(src,int(src.shape[-1]*ap/a.phase_randomise),a.rng)
    # Original uses an else-if chain for this group of operations.
    if a.multiply_constant!=1: y*=a.multiply_constant
    elif a.divide_constant!=1:
        if a.divide_constant==0: fail('Cannot divide by zero')
        y/=a.divide_constant
    elif a.add_constant!=0: y+=a.add_constant
    elif a.subtract_constant!=0: y-=a.subtract_constant
    elif any((a.multiply,a.divide,a.add,a.subtract)):
        key=next(k for k in ('multiply','divide','add','subtract') if getattr(a,k)); path=getattr(a,key)
        other,_=read_mrc(path)
        if other.shape!=y.shape: fail('Image arithmetic inputs have different shapes')
        if key=='multiply': y*=other
        elif key=='divide': y=np.divide(y,other,out=np.zeros_like(y),where=np.abs(other)>=1e-10)
        elif key=='add': y+=other
        else:
            scale=1.
            if a.optimise_scale_subtract:
                mask=np.ones(y.shape) if not a.mask_optimise_subtract else read_mrc(a.mask_optimise_subtract)[0]
                if mask.shape!=y.shape: fail('Subtraction mask shape mismatch')
                candidates=np.arange(-a.optimise_bfactor_subtract,a.optimise_bfactor_subtract+1e-6,10) if a.optimise_bfactor_subtract>0 else [0.]
                best=None
                for b in candidates:
                    z=np.fft.irfftn(np.fft.rfftn(other)*b_weights(other.shape,ap,b),s=other.shape) if b else other
                    den=np.sum((mask*z)**2)
                    if den<=0: fail('No signal in subtraction reference/mask')
                    sc=np.sum(mask*mask*src*z)/den; loss=np.sum((mask*(src-sc*z))**2)
                    if best is None or loss<best[0]: best=(loss,sc,b,z)
                _,scale,b,other=best; print('Subtraction scale:',fmt(scale),'B:',fmt(b))
            y-=scale*other
    elif a.HistGram:
        nb=src.shape[-2]; lo=float(src.min()); step=(float(src.max())-lo)/nb
        idx=np.clip(round_rel((src-lo)/step),0,nb-1) if step else np.zeros(src.shape,int)
        counts=np.bincount(idx.ravel(),minlength=nb)*100/src.size
        print('value percentage differential')
        for i,c in enumerate(counts): print(fmt(lo+i*step),fmt(c),fmt(c-counts[i-1] if i else 0))
    elif a.fsc:
        other,_=read_mrc(a.fsc); print(curve_table({'rlnFourierShellCorrelation':fsc(y,other)},y.shape[-1],ap).text())
    elif a.power:
        f=np.fft.rfftn(y)/y.size
        print(curve_table({'rlnReferenceSpectralPower':shell_mean(np.abs(f)**2,y.shape,n=y.shape[-1]//2+2)},y.shape[-1],ap,'power').text())
    elif a.adjust_power:
        other,_=read_mrc(a.adjust_power)
        if other.shape!=src.shape: fail('Power spectrum reference shape mismatch')
        f=np.fft.rfftn(src); tar=shell_mean(np.abs(np.fft.rfftn(other)),src.shape,n=src.shape[-1]); own=shell_mean(np.abs(f),src.shape,n=src.shape[-1])
        ratio=np.divide(tar,own,out=np.zeros_like(tar),where=own>0); y=np.fft.irfftn(f*radial_values(ratio,src.shape),s=src.shape)
    elif a.cos_dphi:
        other,_=read_mrc(a.cos_dphi)
        if other.shape!=src.shape: fail('Phase comparison shape mismatch')
        v=shell_mean(np.cos(np.angle(np.fft.rfftn(y))-np.angle(np.fft.rfftn(other))),y.shape)
        print(curve_table({'rlnFourierShellCorrelation':v},y.shape[-1],ap,'cos').text())
    elif a.correct_avg_ampl:
        amp,_=read_mrc(a.correct_avg_ampl); f=np.fft.rfftn(src)
        if amp.shape!=f.shape: fail('Average amplitude reference must have half-complex shape, as produced by --avg_ampl')
        rad=radial_values(shell_mean(amp,src.shape,n=src.shape[-1]),src.shape)
        corr=np.divide(amp,rad,out=np.ones_like(amp),where=rad!=0); corr[radius(src.shape,True)<=a.minr_ampl_corr]=1
        y=np.fft.irfftn(np.divide(f,corr,out=np.zeros_like(f),where=np.abs(corr)>1e-20),s=src.shape)
    elif a.fourier_filter:
        filt,_=read_mrc(a.fourier_filter)
        if filt.shape!=src.shape: fail('Fourier filter must have the same full shape as input, DC at box center')
        gs=grids(src.shape,True); coords=list(np.broadcast_arrays(*gs)); theta=np.deg2rad(psi)
        xx=coords[-1].copy(); yy=coords[-2].copy()
        coords[-1]=round_rel(xx*np.cos(theta)+yy*np.sin(theta)); coords[-2]=round_rel(-xx*np.sin(theta)+yy*np.cos(theta))
        coords=np.array([v+n//2 for v,n in zip(coords,src.shape)])
        w=ndi.map_coordinates(filt,coords,order=0,mode='constant',cval=0,prefilter=False)
        y=np.fft.irfftn(np.fft.rfftn(src)*w,s=src.shape)
    elif a.fsc_weighting:
        c=np.maximum(read_curve(a.fsc_weighting),0)
        # SIRM v1p2's option multiplies by FSC directly, not sqrt(2 FSC/(1+FSC)).
        y=np.fft.irfftn(np.fft.rfftn(src)*radial_values(c,src.shape,outside=1),s=src.shape)
    elif a.test_F:
        if src.ndim!=3 or a.C_OUT<2: fail('--test_F requires a volume and C_OUT>=2')
        tmp=resize_fourier(window(src,src.shape[-1]*2),a.C_OUT+a.C_OUT%2); tmp=window(tmp,a.C_OUT)
        tmp=ndi.shift(tmp,[a.C_OUT/2,a.C_OUT/2,-a.C_OUT/2],order=1,mode='grid-wrap',prefilter=False)
        write_mrc('debug_BB.mrc',tmp,ap)
    elif a.normal_background_powerspectrum:
        if src.ndim!=2 or src.shape!=(a.boxsize,a.boxsize): fail('--boxsize must match square 2D input for background whitening')
        bg=radius(src.shape)>a.bg_radius
        if not bg.any(): fail('Background radius leaves no background pixels')
        mask=automask(bg.astype(float),.5,0,12 if a.bg_radius>=.5 else 0)
        spec=shell_mean(np.abs(np.fft.rfftn(y*mask))/y.size,y.shape,n=y.shape[-1])
        w=radial_values(spec,y.shape)
        y=np.fft.irfftn(np.divide(np.fft.rfftn(y),w,out=np.zeros_like(w,dtype=complex),where=w>1e-20),s=y.shape)
        # RELION normalization removes a fitted background plane before sigma scaling.
        gy,gx=np.broadcast_arrays(*grids(y.shape)); design=np.column_stack((gx[bg],gy[bg],np.ones(bg.sum())))
        coeff=np.linalg.lstsq(design,y[bg],rcond=None)[0]; y-=coeff[0]*gx+coeff[1]*gy+coeff[2]
        sd=y[bg].std(ddof=0)
        if sd==0: fail('Background standard deviation is zero')
        y=(y-y[bg].mean())/sd
    if a.bfactor: y=np.fft.irfftn(np.fft.rfftn(y)*b_weights(y.shape,ap,a.bfactor),s=y.shape)
    if a.LoG>0:
        oldshape=y.shape
        if len(set(y.shape))!=1:
            if y.ndim!=2: fail('LoG requires a cubic volume')
            y=pad_noise(y,a.rng)
        r=radius(y.shape,True); isigma2=(.5*y.shape[-1]*ap/a.LoG)**2
        y=np.fft.irfftn(np.fft.rfftn(y)*(r*r/isigma2)*np.exp(-.5*r*r/isigma2),s=y.shape)
        if y.shape!=oldshape: y=window(y,oldshape)
    if a.lowpass>0: y=filter_map(y,ap,a.lowpass,a.filter_edge_width,direction=a.directional,rng=a.rng)
    if a.highpass>0: y=filter_map(y,ap,a.highpass,a.filter_edge_width,True)
    # Source reads the original input for mirror operations, overriding earlier filters.
    if a.flipX: y=src[...,::-1].copy()
    elif a.flipY: y=np.flip(src,-2).copy()
    elif a.flipZ:
        if src.ndim!=3: fail('--flipZ requires a 3D map')
        y=src[::-1].copy()
    elif a.invert_hand: y=np.take(src,(-np.arange(src.shape[-1]))%src.shape[-1],axis=-1)
    shift=[a.shift_z,a.shift_y,a.shift_x][-y.ndim:]
    if a.shift_com:
        positive=np.maximum(y,0); total=positive.sum()
        if total==0: fail('No positive signal for --shift_com')
        shift=[-float(np.sum(positive*g)/total) for g in grids(y.shape)]
    if any(shift): y=ndi.shift(y,shift,order=1,mode='constant',cval=0,prefilter=False)
    if a.rescale_angpix>0:
        shape=y.shape; old=max(shape)
        if len(set(shape))!=1:
            if y.ndim!=2: fail('Rescaling a noncubic volume is not supported')
            y=window(y,old)
        new=int(math.floor(old*ap/a.rescale_angpix+.5)); new-=new%2
        if new<2: fail('Requested pixel size gives a box smaller than 2 pixels')
        real=old*ap/new; y=resize_fourier(y,new)
        if len(set(shape))!=1:
            sh=[int(math.floor(n*ap/real+.5)) for n in shape]; sh=[n-n%2 for n in sh]; y=window(y,sh)
        ap=real
    if a.meanshrink>1:
        # Preserve source ROUND-based accumulation; reject out-of-box samples instead of corrupting memory.
        sh=tuple(int(math.floor(n/a.meanshrink+.5)) for n in src.shape)
        if min(sh)<1: fail('meanshrink is too large')
        coords=np.indices(src.shape); dest=round_rel(coords/a.meanshrink); valid=np.ones(src.shape,bool)
        for d,n in zip(dest,sh): valid &= d<n
        z=np.zeros(sh); np.add.at(z,tuple(d[valid] for d in dest),src[valid]/a.meanshrink**src.ndim); y=z
    if a.new_box>0: y=window(y,a.new_box)
    if a.sym:
        if y.ndim!=3: fail('--sym requires 3D')
        y=symmetrize(y,a.sym)
    if a.threshold_above!=999: y=np.minimum(y,a.threshold_above)
    if a.threshold_below!=-999: y=np.maximum(y,a.threshold_below)
    if a.force_header_angpix>0: ap=a.force_header_angpix
    return y,ap

def write_png(path,x,a):
    import struct,zlib
    if x.ndim!=2: fail('PNG output requires a single 2D image')
    lo,hi=a.black,a.white
    if lo==hi or a.sigma_contrast>0:
        if a.sigma_contrast>0: lo=x.mean()-a.sigma_contrast*x.std(ddof=0); hi=x.mean()+a.sigma_contrast*x.std(ddof=0)
        else: lo,hi=float(x.min()),float(x.max())
    u=np.clip((x-lo)/(hi-lo if hi>lo else 1),0,1)
    grey=np.floor(u*255).astype(int); red=green=blue=grey
    if a.colour_fire:
        red=np.where(grey>=128,255,grey*2); green=blue=np.where(grey>=128,2*(255-grey),grey*2)
    elif a.colour_ice:
        red=np.where(grey>=128,2*(grey-128),0); green=blue=np.where(grey>=128,2*(grey-128),255-2*grey)
    elif a.colour_fire_n_ice:
        group=grey//85; frac=np.floor(255*(grey/85-group)).astype(int)
        red=np.select([group==0,group==1],[0,frac],default=255)
        green=blue=np.select([group==0,group==1,group==2],[255-frac,frac,255-frac],default=0)
    elif a.colour_rainbow:
        q=(255-grey)/64; group=np.floor(q).astype(int); frac=np.floor(255*(q-group)).astype(int)
        red=np.select([group==0,group==1],[255,255-frac],default=0)
        green=np.select([group==0,(group==1)|(group==2),group==3],[frac,255,255-frac],default=0)
        blue=np.select([group<2,group==2],[0,frac],default=255)
    elif a.colour_difference:
        red=np.clip(3*(grey-128),0,255); green=np.clip(3*(np.abs(grey-128)-42),0,255); blue=np.clip(-3*(grey-128),0,255)
    arr=np.stack([red,green,blue],axis=-1).astype('uint8'); color=2
    def chunk(name,data): return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in arr)
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',x.shape[1],x.shape[0],8,color,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_bytes(data)

def generate_mp(x,a):
    if x.ndim!=3 or not a.MP_table or a.MP_stepsize<=0: fail('--gen_MP_table requires 3D input, --MP_table and positive --MP_stepsize')
    entries=[[0.,0.]]; mats=[euler_matrix(0,0,0)]
    for loc in np.argwhere(x>.5):
        v=(loc-np.array(x.shape)//2)[::-1].astype(float); norm=np.linalg.norm(v)
        if not norm: continue
        v/=norm; rt=[np.degrees(np.arctan2(v[1],v[0])),np.degrees(np.arccos(np.clip(v[2],-1,1)))]; m=euler_matrix(*rt,0)
        dist=np.degrees(np.arccos(np.clip(np.sum(np.array(mats)*m,axis=2),-1,1))).mean(axis=1)
        if np.all(dist>=a.MP_stepsize): entries.append(rt); mats.append(m)
    star_table('',['rlnMaskPickingRot','rlnMaskPickingTilt'],entries).write(a.MP_table)
    print('Mask-picking directions:',len(entries))

def main(a):
    numerical(any((a.sym,a.shift_x,a.shift_y,a.shift_z,a.shift_com,a.fourier_filter,a.test_F,a.normal_background_powerspectrum,a.avg_ampl2_ali)))
    a.rng=RelionRNG(a.random_seed)
    report=a.stats or a.com or a.fsc or a.power or a.cos_dphi
    if not a.o and not report and not a.gen_MP_table: fail('Specify --o')
    if a.filter_edge_width<0: fail('Filter width cannot be negative')
    s=None; tab=None; items=[]
    if a.i.lower().endswith('.star'):
        s=Star.read(a.i); _,tab=s.table(); key='rlnMicrographMovieName' if a.average_all_movie_frames else 'rlnImageName'
        ci=tab.col(key)
        items=[(resolve_image(r[ci],a.i,a.data_root),r) for r in tab.rows]
    else:
        name=resolve_image(a.i,None,a.data_root)
        if '@' not in name and name.lower().endswith('.mrcs') and not (a.avg_bin>0 or (a.avg_first>=0 and a.avg_last>=0) or a.average_all_movie_frames):
            with mrcfile.mmap(name,mode='r') as m: n=m.data.shape[0] if m.data.ndim==3 else 1
            items=[(f'{i+1}@{name}',None) for i in range(n)]
        else: items=[(name,None)]
    if not items: fail('No input images')
    indexed_output='@' in a.o
    if indexed_output and (len(items)!=1 or s is not None): fail('Indexed output accepts one image, without STAR input')
    average=any((a.average,a.avg_ampl,a.avg_ampl2,a.avg_ampl2_ali,a.average_all_movie_frames))
    movie=a.avg_bin>0 or (a.avg_first>=0 and a.avg_last>=0)
    if movie and len(items)>1: fail('Movie bin/range processing accepts one movie per invocation')
    stackout=bool(a.o.lower().endswith('.mrcs')) and not average and not movie and not indexed_output
    outputs={}; targets=[]
    if a.o and not average and not movie:
        counts={}
        for name,row in items:
            if stackout: dest=a.o
            elif s is not None or len(items)>1:
                if Path(a.o).name!=a.o: fail('For STAR/stack input use a plain suffix or an output .mrcs path')
                dest=suffix(name.split('@')[-1],'_'+a.o)
            else: dest=a.o
            if Path(dest.split('@')[-1]).resolve()==Path(name.split('@')[-1]).resolve(): fail('Output must differ from input')
            index=counts.get(dest,0); counts[dest]=index+1; targets.append((dest,index))
        totals=counts
    else: totals={}
    total=None; total_n=0; lastmeta=None; optics_updates={}
    try:
        for i,(name,row) in enumerate(items):
            x,meta=read_mrc(name); ap=a.angpix if a.angpix>0 else meta['angpix']; lastmeta=meta
            if s is not None and a.angpix<=0: ap=pixel_for(s,tab,row,ap)
            if a.stats:
                print(f'{name} : shape={x.shape} avg= {x.mean():.9g} stddev= {x.std(ddof=0):.9g} minval= {x.min():.9g} maxval= {x.max():.9g}; angpix = {meta["angpix"]:.9g}'); continue
            if a.com:
                w=np.maximum(x,0); mass=w.sum(); com=[float((w*g).sum()/mass) if mass else 0 for g in grids(x.shape)]
                print(name,': center of mass (x,y,z):',list(reversed(com))); continue
            if a.gen_MP_table: generate_mp(x,a)
            if movie:
                if x.ndim!=3 or not meta['is_stack']: fail('Movie operations require .mrcs stack')
                if a.avg_bin>0:
                    n=len(x)//a.avg_bin
                    if n==0: fail('avg_bin exceeds number of frames')
                    y=x[:n*a.avg_bin].reshape(n,a.avg_bin,*x.shape[1:]).sum(axis=1)
                    if n==1: y=y[0]
                else:
                    if not 1<=a.avg_first<=a.avg_last<=len(x): fail('Frame range is one based and must be inside movie')
                    y=x[a.avg_first-1:a.avg_last].sum(axis=0)
                write_mrc(a.o,y,ap,meta,stack=y.ndim==3); continue
            if average:
                y=x
                n=1
                if a.average_all_movie_frames:
                    if x.ndim!=3 or not meta['is_stack']: fail('Movie averaging expects stacks')
                    y=x.sum(axis=0); n=len(x)
                elif a.avg_ampl or a.avg_ampl2 or a.avg_ampl2_ali:
                    if a.avg_ampl2_ali:
                        if tab is None: fail('--avg_ampl2_ali requires STAR orientations')
                        psi=float(tab.get(row,'rlnAnglePsi',0)); theta=np.deg2rad(psi)
                        mat=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
                        shift=np.array([float(tab.get(row,'rlnOriginY',0)),float(tab.get(row,'rlnOriginX',0))]); center=np.array(x.shape)//2
                        y=ndi.affine_transform(x,mat.T,offset=center-mat.T@(center+shift),order=1,mode='constant',prefilter=False)
                    y=np.abs(np.fft.rfftn(y))/y.size
                    if not a.avg_ampl: y=y*y
                if total is None: total=np.zeros_like(y)
                if total.shape!=y.shape: fail('Cannot average different image shapes')
                total+=y; total_n+=n; continue
            psi=float(tab.get(row,'rlnAnglePsi',0)) if tab else 0
            y,outap=transform_image(x,ap,a,psi)
            if a.o:
                dest,idx=targets[i]
                if dest.lower().endswith('.png'): write_png(dest,y,a)
                elif stackout or totals[dest]>1 or ('@' in name and (s is not None or len(items)>1)):
                    if y.ndim!=2: fail('Output particle stacks require 2D particles')
                    if dest not in outputs:
                        Path(dest).parent.mkdir(parents=True,exist_ok=True)
                        mm=mrcfile.new_mmap(dest,shape=(totals[dest],*y.shape),mrc_mode=2,overwrite=True); mm.set_image_stack(); mm.voxel_size=outap; outputs[dest]=mm
                    mm=outputs[dest]
                    if mm.data.shape[1:]!=y.shape or not math.isclose(float(mm.voxel_size.x),outap,rel_tol=1e-6): fail('Cannot mix image sizes/pixel sizes within one output stack')
                    mm.data[idx]=y
                else: write_mrc(dest,y,outap,meta)
                if row is not None:
                    row[ci]=(f'{idx+1:06d}@' if dest in outputs else '')+str(Path(dest).resolve()).replace('\\','/')
                    group=tab.get(row,'rlnOpticsGroup','1'); optics_updates[group]=(y.shape[-1],outap)
            if len(items)>10 and (i+1)%100==0: print(f'Processed {i+1}/{len(items)}',flush=True)
    finally:
        for mm in outputs.values(): mm.update_header_stats(); mm.close()
    if average:
        if Path(a.o).resolve()==Path(items[0][0].split('@')[-1]).resolve(): fail('Output must differ from input')
        write_mrc(a.o,total/total_n,ap,lastmeta)
    if s is not None and a.o and not average and not movie and not (a.stats or a.com):
        for b,t in s.tables():
            if b.name=='optics':
                for row in t.rows:
                    group=t.get(row,'rlnOpticsGroup')
                    if group in optics_updates:
                        size,ap=optics_updates[group]; row[t.add('rlnImageSize')]=str(size); row[t.add('rlnImagePixelSize')]=fmt(ap)
        outstar=str(Path(a.o).with_suffix('.star')) if stackout else suffix(a.i,'_'+a.o)
        if Path(outstar).resolve()==Path(a.i).resolve(): fail('Output STAR would overwrite input')
        s.write(outstar); print('Written STAR:',outstar)
    if a.o: print('Output:',a.o)

if __name__=='__main__': entrypoint()
