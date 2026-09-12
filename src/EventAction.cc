#include "EventAction.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

#include <fstream>
#include <iomanip>

EventAction::EventAction(
    G4double momentum,
    G4double refractiveIndex)
    : fMomentum(momentum),
      fRefractiveIndex(refractiveIndex)
{
}

EventAction::~EventAction()
{
}

void EventAction::BeginOfEventAction(const G4Event*)
{
    fHits.clear();
}

void EventAction::AddHit(
    G4double x,
    G4double y,
    G4double energy)
{
    fHits.push_back({x, y, energy});
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    std::ofstream file("hits.csv", std::ios::app);

    file << event->GetEventID()
         << ",pi+,"
         << fMomentum / GeV << ","
         << fRefractiveIndex
         << ",\"[";

    for (size_t i = 0; i < fHits.size(); ++i)
    {
        if (i > 0)
            file << ",";

        file << "("
             << std::setprecision(8)
             << fHits[i].x / mm << ","
             << fHits[i].y / mm << ","
             << fHits[i].energy / eV
             << ")";
    }

    file << "]\"\n";

    file.close();
}