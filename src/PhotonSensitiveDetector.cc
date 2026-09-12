#include "PhotonSensitiveDetector.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4EventManager.hh"
#include "G4UserEventAction.hh"
#include "G4SystemOfUnits.hh"

#include "EventAction.hh"

PhotonSensitiveDetector::PhotonSensitiveDetector(
    const G4String& name)
    : G4VSensitiveDetector(name)
{
}

PhotonSensitiveDetector::~PhotonSensitiveDetector()
{
}

G4bool PhotonSensitiveDetector::ProcessHits(
    G4Step* step,
    G4TouchableHistory*)
{
    G4Track* track = step->GetTrack();

    if (track->GetDefinition() !=
        G4OpticalPhoton::Definition())
        return false;

    G4ThreeVector position =
        step->GetPreStepPoint()->GetPosition();

    G4double energy =
        track->GetTotalEnergy();

    auto* eventAction =
        static_cast<EventAction*>(
            G4EventManager::GetEventManager()
                ->GetUserEventAction());

    eventAction->AddHit(
        position.x(),
        position.y(),
        energy);

    track->SetTrackStatus(fStopAndKill);

    return true;
}